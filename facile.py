#!/usr/bin/env python3

import argparse
import importlib
import math
import sys
from collections import namedtuple
from itertools import count, cycle
from typing import List

from instructions import *
from microArchConfigs import MicroArchConfig, MicroArchConfigs
from utils import *
from x64_lib import *


def computePortUsageLimit(instructions, instrInstancesForInstr={}):
   portUsage = {}
   for instr in instructions:
      if instr.macroFusedWithPrevInstr:
         continue
      for ports, nUops in instr.portData.items():
         if instr.mayBeEliminated:
            instrIList = instrInstancesForInstr.get(instr, [])
            if instrIList:
               nUops = sum((not uop.eliminated) for instrI in instrIList for lamUop in instrI.uops for uop in lamUop.getUnfusedUops())/len(instrIList)
            else:
               continue
         portUsage[frozenset(ports)] = portUsage.get(frozenset(ports), 0) + nUops

   TP = 0
   for pc in set(pc|pc2 for pc in portUsage for pc2 in portUsage):
      uops = sum(u for pc2, u in portUsage.items() if pc2.issubset(pc))
      TP = max(TP, uops/len(pc))
   return TP


def computeIssueLimit(instructions: List['Instr'], uArchConfig: 'MicroArchConfig'):
   return sum(i.retireSlots for i in instructions if not i.macroFusedWithPrevInstr)/uArchConfig.issueWidth


def hasLCP(instrD):
   return (instrD['prefix66'] != '0') and (instrD.get('IMM_WIDTH', 0) == 16)


def computePredecLimit(disas, loop=False, alignmentOffset=0):
   codeLength = sum(len(d['opcode']) for d in disas) // 2
   unroll = 1 if loop else (16 // math.gcd(codeLength, 16))
   nB16Blocks = int(math.ceil((unroll * codeLength) / 16))
   L = [0 for _ in range(0, nB16Blocks)] # number of instr. instances whose last byte is in the given block
   O = [0 for _ in range(0, nB16Blocks)] # number of instr. instances whose nominal opcode starts in the given block but whose last byte is in the next block
   LCP = [0 for _ in range(0, nB16Blocks)] # number of instr. instances whose nominal opcode starts in the given block, and which have a length-changing prefix
   alignmentOffset = alignmentOffset % 16
   curAddr = (-16 + alignmentOffset) if alignmentOffset else 0
   for d in cycle(disas):
      if curAddr >= unroll * codeLength:
         break

      nextAddr = curAddr + (len(d['opcode']) // 2)
      endBlock = (nextAddr-1) // 16 # 16-Byte block in which the last Byte of the instruction is stored
      posNominalOpcode = d['pos_nominal_opcode']
      nominalOpcodeBlock = (curAddr + posNominalOpcode) // 16
      curAddr = nextAddr

      if 0 <= endBlock < nB16Blocks:
         L[endBlock] += 1
      if 0 <= nominalOpcodeBlock < nB16Blocks:
         if nominalOpcodeBlock != endBlock:
            O[nominalOpcodeBlock] += 1
         if hasLCP(d):
            LCP[nominalOpcodeBlock] += 1

   cycles = 0
   for bi in range(0, nB16Blocks):
      cycles += math.ceil((L[bi]+O[bi])/5)
      cycles += max(0, 3 * LCP[bi] - (math.ceil((L[bi-1]+O[bi-1])/5) - 1))
   return cycles / unroll


def computePredecLimitSimple(hex, instructions):
   codeLength = len(hex) // 2
   return codeLength/16


def computeDecLimit(instructions, uArchConfig):
   instructions = [i for i in instructions if not i.macroFusedWithPrevInstr]
   firstInstrOnDecInRound = {}
   nAvailSimpleDec = 0
   curDec = uArchConfig.nDecoders - 1
   nComplexDecInRound = {}
   for round in count(0):
      nComplexDecInRound[round] = 0
      for ii, instr in enumerate(instructions):
         if instr.complexDecoder:
            curDec = 0
            nAvailSimpleDec = instr.nAvailableSimpleDecoders
         else:
            if ((nAvailSimpleDec == 0)
                  or (curDec+1 == uArchConfig.nDecoders-1 and instr.macroFusibleWith and (not uArchConfig.macroFusibleInstrCanBeDecodedAsLastInstr))):
               curDec = 0
               nAvailSimpleDec = uArchConfig.nDecoders - 1
            else:
               curDec += 1
               nAvailSimpleDec -= 1
         if instr.isBranchInstr or instr.macroFusedWithNextInstr:
            nAvailSimpleDec = 0

         if curDec == 0:
            nComplexDecInRound[round] += 1

         if ii == 0:
            if curDec in firstInstrOnDecInRound:
               firstRound = firstInstrOnDecInRound[curDec]
               return sum(nComplexDecInRound[r] for r in range(firstRound, round)) / (round - firstRound)
            else:
               firstInstrOnDecInRound[curDec] = round


def computeDecLimitSimple(instructions):
   instructions = [i for i in instructions if not i.macroFusedWithPrevInstr]
   return max(len(instructions)/4, len([i for i in instructions if i.complexDecoder]))


def computeLSDLimit(instructions, uArchConfig):
   nUops = sum(i.uopsMITE + i.uopsMS for i in instructions if not i.macroFusedWithPrevInstr)
   LSDUnrollCount = uArchConfig.LSDUnrolling.get(nUops, 1)
   return math.ceil((nUops * LSDUnrollCount) / uArchConfig.issueWidth) / LSDUnrollCount


def computeDSBLimit(instructions, alignmentOffset=0):
   nUops = sum(i.uopsMITE for i in instructions if not i.macroFusedWithPrevInstr)
   codeLength = sum(len(i.opcode) // 2 for i in instructions[:-1])
   if (codeLength + alignmentOffset) // 32 == alignmentOffset // 32:
      return math.ceil(nUops/6)
   else:
      return nUops/6


LatGraphEdge = namedtuple('LatencyGraphEdge', ['source', 'target', 'cost', 'time'])
def generateLatencyGraph(instructions: List[Instr], uArchConfig: MicroArchConfig, initPolicy):
   moves = [instr for instr in instructions if instr.mayBeEliminated]
   prevWriteForMove = {}
   prevNonEliminatedWriteForMove = {}
   outputOfMoveRenamedBy32BitMove = {}

   prevWriteToReg = {}
   for instr in instructions * 2:
      if instr.mayBeEliminated:
         prevWriteForMove[instr] = prevWriteToReg.get(getCanonicalReg(instr.inputRegOperands[0].reg))
      for outOp in instr.outputRegOperands:
         prevWriteToReg[getCanonicalReg(outOp.reg)] = instr

   for move in moves:
      movesOnPath = set()
      curInstr = move
      while (curInstr is not None) and (curInstr not in movesOnPath):
         if curInstr.mayBeEliminated and (getRegSize(curInstr.outputRegOperands[0].reg) == 32):
            outputOfMoveRenamedBy32BitMove[move] = True
         if not curInstr.mayBeEliminated:
            prevNonEliminatedWriteForMove[move] = curInstr
            break
         movesOnPath.add(curInstr)
         curInstr = prevWriteForMove[curInstr]

   prevWriteToKey = dict() # key -> (instr, outOp, fastPtrChasing, iteration)
   absValGen = AbstractValueGenerator(initPolicy)

   def getOpKey(op):
      if isinstance(op, RegOperand):
         return getCanonicalReg(op.reg)
      elif isinstance(op, FlagOperand):
         return op.flags
      elif isinstance(op, MemOperand):
         memAddr = op.memAddr
         return (absValGen.getAbstractValueForReg(memAddr.get('base')), absValGen.getAbstractValueForReg(memAddr.get('index')),
                  memAddr.get('scale'), memAddr.get('disp'))
      else:
         return None

   RSPImplicitlyChanged = False
   def processInstrOutputs(instr: Instr, iteration):
      fastPtrChasing = False
      if uArchConfig.fastPointerChasing and instr.inputMemOperands:
         baseReg = instr.inputMemOperands[0].memAddr.get('base')
         baseInstr = ((prevNonEliminatedWriteForMove.get(prevWriteToKey[baseReg][0]) if prevWriteToKey[baseReg][0].mayBeEliminated else prevWriteToKey[baseReg][0])
                        if baseReg in prevWriteToKey else None)
         baseRenamedBy32BitMove = (baseReg in prevWriteToKey) and outputOfMoveRenamedBy32BitMove.get(prevWriteToKey[baseReg][0])
         indexReg = instr.inputMemOperands[0].memAddr.get('index')
         indexInstr = ((prevNonEliminatedWriteForMove.get(prevWriteToKey[indexReg][0]) if prevWriteToKey[indexReg][0].mayBeEliminated else prevWriteToKey[indexReg][0])
                        if indexReg in prevWriteToKey else None)

         fastPtrChasing = latReducedDueToFastPtrChasing(uArchConfig, instr.inputMemOperands[0].memAddr, baseInstr, indexInstr, baseRenamedBy32BitMove)

      nonlocal RSPImplicitlyChanged
      if instr.implicitRSPChange:
         RSPImplicitlyChanged = True
      elif any((getCanonicalReg(op.reg) == 'RSP') for op in instr.inputRegOperands+instr.memAddrOperands+instr.outputRegOperands):
         RSPImplicitlyChanged = False

      for outOp in instr.outputRegOperands + instr.outputFlagOperands + instr.outputMemOperands:
         prevWriteToKey[getOpKey(outOp)] = (instr, outOp, fastPtrChasing, iteration)
         if isinstance(outOp, RegOperand):
            absValGen.setAbstractValueForCurInstr(getOpKey(outOp), instr)

      absValGen.finishCurInstr()

   for instr in instructions * 2:
      processInstrOutputs(instr, 0)

   nodesForInstr = {}
   edgesForNode = {}

   for instr in instructions:
      nodesForInstr[instr] = []
      for op in instr.inputRegOperands + instr.memAddrOperands + instr.inputFlagOperands + instr.inputMemOperands:
         nodesForInstr[instr].append(op)

         if getOpKey(op) in prevWriteToKey:
            prevInstr, prevOutOp, fastPtrChasing, prevIt = prevWriteToKey[getOpKey(op)]
            for prevInOp in prevInstr.inputRegOperands + prevInstr.memAddrOperands + prevInstr.inputFlagOperands + prevInstr.inputMemOperands:
               if prevInstr.mayBeEliminated:
                  lat = 0
               elif (not isinstance(prevInOp, MemOperand)) and isinstance(op, MemOperand):
                  lat = 0 # ToDo
               elif prevInstr.latencies.get((prevInOp, prevOutOp), 0) > 0:
                  lat = prevInstr.latencies[(prevInOp, prevOutOp)]
                  if fastPtrChasing and (prevInOp in prevInstr.memAddrOperands) and prevInstr.inputMemOperands:
                     lat -= 1
                  elif (RSPImplicitlyChanged and (prevInOp in instr.inputRegOperands+instr.memAddrOperands)
                        and (getCanonicalReg(prevInOp.reg) == 'RSP') and (not prevInOp.isImplicitStackOperand)):
                     lat += 1
               else:
                  continue

               edge = LatGraphEdge(prevInOp, op, lat, (0 if prevIt else 1))
               edgesForNode.setdefault(prevInOp, []).append(edge)

      processInstrOutputs(instr, 1)
   return (nodesForInstr, edgesForNode)


def computeMaximumLatencyForGraph(instructions: List[Instr], nodesForInstr, edgesForNode):
   # based on https://stackoverflow.com/a/62006383/10461973
   def findStronglyConnectedComponents(nodesForInstr, edgesForNode):
      indexDict = {}
      lowlinkDict = {}
      onStackSet = set()
      S = []
      components = []

      for nodeList in nodesForInstr.values():
         for n in nodeList:
            if n not in indexDict:
               callStack  = [(n, 0)]

               while (callStack):
                  v, pi = callStack.pop()

                  if pi == 0:
                     index = len(indexDict)
                     lowlinkDict[v] = index
                     indexDict[v] = index
                     S.append(v)
                     onStackSet.add(v)
                  else:
                     prev = edgesForNode[v][pi-1].target
                     lowlinkDict[v] = min(lowlinkDict[v], lowlinkDict[prev])

                  while pi < len(edgesForNode.get(v, [])) and (edgesForNode[v][pi].target in indexDict):
                     w = edgesForNode[v][pi].target
                     if w in onStackSet:
                        lowlinkDict[v] = min(lowlinkDict[v], indexDict[w])
                     pi += 1

                  if pi < len(edgesForNode.get(v, [])):
                     w = edgesForNode[v][pi].target
                     callStack.append((v, pi+1))
                     callStack.append((w, 0))
                     continue

                  if lowlinkDict[v] == indexDict[v]:
                     comp = []
                     while True:
                        w = S.pop()
                        onStackSet.remove(w)
                        comp.append(w)
                        if v == w:
                           break
                     components.append(comp)
      return components

   # based on the "VAL" algorithm described in https://doi.org/10.1145/1027084.1027085
   def maximumCycleRatio(nodes, edges, r=sys.maxsize, eps=0.01):
      def findRatio(nodes, r, p):
         visited = {v: None for v in nodes}
         handle = None
         for v in nodes:
            if visited[v] is not None:
               continue
            u = v
            while True:
               visited[u] = v
               u = p[u].target
               if visited[u] is not None:
                  break
            if visited[u] != v:
               continue
            x = u
            sum = 0
            len = 0
            while True:
               sum = sum - p[u].cost
               len = len + p[u].time
               u = p[u].target
               if x == u:
                  break
            if r > sum/len:
               r = sum/len
               handle = u
         return (r, handle)

      d = {v: sys.maxsize for v in nodes}
      p = {v: None for v in nodes}

      for e in edges:
         if -e.cost < d[e.source]:
            d[e.source] = -e.cost
            p[e.source] = e

      edgesOnMaxCycle = []
      while True:
         r, handle = findRatio(nodes, r, p)
         if handle:
            edgesOnMaxCycle = []
            u = handle
            while True:
               edgesOnMaxCycle.append(p[u])
               u = p[u][1]
               if u == handle:
                  break

         changed = False
         for e in edges:
            if d[e.source] > d[e.target] - e.cost - r*e.time + eps:
               d[e.source] = d[e.target] - e.cost - r*e.time
               p[e.source] = e
               changed = True
         if not changed:
            return (-r, edgesOnMaxCycle)

   components = findStronglyConnectedComponents(nodesForInstr, edgesForNode)

   maxCycleRatio = 0
   edgesOnMaxCycle = []
   for comp in components:
      edgesForComp = [e for n in comp for e in edgesForNode.get(n, []) if e.target in comp]
      if edgesForComp:
         curMaxCycleRatio, curEdgesOnMaxCycle = maximumCycleRatio(comp, edgesForComp)
         if curMaxCycleRatio > maxCycleRatio:
            maxCycleRatio = curMaxCycleRatio
            edgesOnMaxCycle = curEdgesOnMaxCycle

   return (maxCycleRatio, edgesOnMaxCycle, components)


def generateGraphvizOutputForLatencyGraph(instructions: List[Instr], nodesForInstr, edgesForNode, edgesOnMaxCycle, stronglyConnectedComponents, filename):
   import pydot
   graph = pydot.Dot("g", graph_type="digraph", bgcolor="white")

   for i, instr in enumerate(instructions):
      cluster = pydot.Cluster(str(id(instr)), label=str(i) + ': ' + instr.asm)
      graph.add_subgraph(cluster)

      prevNodeId = None
      for node in nodesForInstr[instr]:
         label = ''
         shape = ''
         fillcolor = 'aqua'
         color = 'black'
         penwidth = 1
         if isinstance(node, RegOperand):
            label = node.reg
            if node in instr.memAddrOperands:
               shape = 'hexagon'
               fillcolor = 'darkolivegreen1'
            else:
               shape = 'oval'
               fillcolor = 'darkslategray1'
         elif isinstance(node, MemOperand):
            label = 'Mem'
            shape = 'rect'
            fillcolor = 'darksalmon'
         elif isinstance(node, FlagOperand):
            label = node.flags
            shape = 'octagon'
            fillcolor = 'gold'
         if any(node == e.source for e in edgesOnMaxCycle):
            color = 'red'
            penwidth = 3
         cluster.add_node(pydot.Node(str(id(node)), label=label, shape=shape, color=color, fillcolor=fillcolor, penwidth=penwidth, style='filled'))
         if prevNodeId:
            graph.add_edge(pydot.Edge(prevNodeId, str(id(node)), style='invis'))
         prevNodeId = str(id(node))

   for nodeList in nodesForInstr.values():
      for node in nodeList:
         for e in edgesForNode.get(node, []):
            color = 'lightgray'
            penwidth = 1
            if e in edgesOnMaxCycle:
               color = 'red'
               penwidth = 3
            elif any(e.source in comp and e.target in comp for comp in stronglyConnectedComponents):
               color = 'blue'

            graph.add_edge(pydot.Edge(str(id(e.source)), str(id(e.target)), xlabel=e.cost, constraint=False, color=color, fontcolor=color,
                           style=('dashed' if e.time else ''), headport='w', tailport='e', penwidth=penwidth))

   graph.write(filename, format=(filename.split('.')[-1] if ('.' in filename) else 'dot'), prog='dot')


def getAnalyticalPredictionForUnrolling(instructions: List[Instr], hex, xedDisas, uArchConfig: MicroArchConfig, components: List[str]):
   TPs = []
   if 'predec' in components:
      TPs.append(('predec', computePredecLimit(xedDisas)))
   if 'predecSimple' in components:
      TPs.append(('predec', computePredecLimitSimple(hex, instructions)))
   if 'dec' in components:
      TPs.append(('dec', computeDecLimit(instructions, uArchConfig)))
   if 'decSimple' in components:
      TPs.append(('decSimple', computeDecLimitSimple(instructions)))
   if 'issue' in components:
      TPs.append(('issue', computeIssueLimit(instructions, uArchConfig)))
   if 'portUsage' in components:
      TPs.append(('portUsage', computePortUsageLimit(instructions)))
   if 'lat' in components:
      nodesForInstr, edgesForNode = generateLatencyGraph(instructions, uArchConfig, 'stack')
      lat = computeMaximumLatencyForGraph(instructions, nodesForInstr, edgesForNode)[0]
      TPs.append(('lat', lat))

   return TPs


def getAnalyticalPredictionForLoop(instructions: List[Instr], hex, xedDisas, uArchConfig: MicroArchConfig, components: List[str]):
   nonMacroFusedInstructions = [instr for instr in instructions if not instr.macroFusedWithPrevInstr]
   if nonMacroFusedInstructions[-1].cannotBeInDSBDueToJCCErratum:
      uopSource = 'MITE'
   elif uArchConfig.LSDEnabled and sum(instr.uopsMITE for instr in nonMacroFusedInstructions) <= uArchConfig.IDQWidth:
      uopSource = 'LSD'
   else:
      uopSource = 'DSB'

   TPs = []
   if 'dsb' in components:
      TPs.append(('dsb', computeDSBLimit(instructions) if (uopSource == 'DSB') else 0))
   if 'lsd' in components:
      TPs.append(('lsd', computeLSDLimit(instructions, uArchConfig) if (uopSource == 'LSD') else 0))
   if 'predec' in components:
      TPs.append(('predec', computePredecLimit(xedDisas, loop=1) if (uopSource == 'MITE') else 0))
   if 'predecSimple' in components:
      TPs.append(('predec', computePredecLimitSimple(hex, instructions) if (uopSource == 'MITE') else 0))
   if 'dec' in components:
      TPs.append(('dec', computeDecLimit(instructions, uArchConfig) if (uopSource == 'MITE') else 0))
   if 'decSimple' in components:
      TPs.append(('decSimple', computeDecLimitSimple(instructions) if (uopSource == 'MITE') else 0))
   if 'issue' in components:
      TPs.append(('issue', computeIssueLimit(instructions, uArchConfig)))
   if 'lat' in components:
      nodesForInstr, edgesForNode = generateLatencyGraph(instructions, uArchConfig, 'stack')
      lat = computeMaximumLatencyForGraph(instructions, nodesForInstr, edgesForNode)[0]
      TPs.append(('lat', lat))
   if 'portUsage' in components:
      TPs.append(('portUsage', computePortUsageLimit(instructions)))

   return TPs


def main():
   parser = argparse.ArgumentParser(description='AvgError')
   parser.add_argument('-hex', type=str, help='Hex code of a basic block')
   parser.add_argument('-file', type=str, help='File with hex codes (one per line)')
   parser.add_argument('-mode', choices=['loop', 'unroll'], required=True)
   parser.add_argument('-arch', help='Microarchitecture', default='SKL')
   parser.add_argument('-analyticalComponents', default='predec,dec,dsb,lsd,issue,portUsage,lat')

   args = parser.parse_args()

   if (args.hex is not None) and (args.file is not None):
      print('-hex and -file are not supported at the same time')
      exit(1)
   if (args.hex is None) and (args.file is None):
      print('either -hex or -file is required')
      exit(1)

   if args.hex is not None:
      lines = [args.hex]
   else:
      with open(args.file, 'r') as f:
         lines = f.read().splitlines()

   import xed
   uArchConfig = MicroArchConfigs[args.arch]
   archData = importlib.import_module('instrData.'+uArchConfig.name+'_data')

   for hex in lines:
      disas = xed.disasHex(hex, chip='TIGER_LAKE')
      instructions = getInstructions(disas, uArchConfig, archData, 0)
      if args.mode == 'unroll':
         TPs = getAnalyticalPredictionForUnrolling(instructions, hex, disas, uArchConfig, args.analyticalComponents.split(','))
      else:
         TPs = getAnalyticalPredictionForLoop(instructions, hex, disas, uArchConfig, args.analyticalComponents.split(','))
      TP = max(v for _, v in TPs)
      print('{}: {:.2f}'.format(hex, TP))


if __name__ == "__main__":
    main()
