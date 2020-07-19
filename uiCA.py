#!/usr/bin/python

import importlib
import os
import random
import re
import sys
from collections import Counter, defaultdict, deque
from itertools import count
from x64_lib import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../XED-to-XML'))
from disas import *

arch = None
clock = 0
allPorts = []
FE_Width = 6
Retire_Width = 4
RB_Width = 224
#nDecoders = 5
Decode_Width = 5

class Uop:
   def __init__(self, instr, rnd, possiblePorts, renamedInputOperands, renamedOutputOperands, latencies):
      self.instr = instr
      self.rnd = rnd # iteration round
      self.possiblePorts = possiblePorts
      self.renamedInputOperands = renamedInputOperands
      self.renamedOutputOperands = renamedOutputOperands
      self.latencies = latencies # latencies[(x,y)] = l; x,y are indices into renamed input and output operands
      self.actualPort = None
      self.allocated = None
      self.dispatched = None
      self.executed = None
      self.retired = None
      self.retireIdx = None # how many other uops were already retired in the same cycle
      self.relatedUops = [self]

   def getUnfusedUops(self):
      return [self]

   def __str__(self):
      return 'UopInstance(rnd: {}, p: {}, a: {}, d: {}, ex: {}, rt: {}, i: {}, o: {})'.format(self.rnd, self.actualPort, self.allocated, self.dispatched, self.executed, self.retired, self.renamedInputOperands, self.renamedOutputOperands)


class FusedUop:
   def __init__(self, uops):
      self.uops = uops

   def getUnfusedUops(self):
      return self.uops


class StackSynchUop(Uop):
   def __init__(self, instr, rnd, renamedInputOperand, renamedOutputOperand):
      possiblePorts = (['0','1','5'] if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB'] else ['0','1','5','6'])
      Uop.__init__(self, instr, rnd, possiblePorts, [renamedInputOperand], [renamedOutputOperand], {(0,0): 1})

# A combination of a StackSynchUop with a (possibly fused) uop that takes only one slot in the decoder; note: does not exist in the actual hardware
class PseudoFusedStackSynchUop: 
   def __init__(self, uops):
      self.uops = uops


class Instr:
   def __init__(self, asm, opcode, nPrefixes, instrStr, portData, uops, retireSlots, inputOperands, indexesOfAddrRegs, outputOperands, latencies, lcpStall,
                modifiesStack):
      self.asm = asm
      self.opcode = opcode
      self.nPrefixes = nPrefixes
      self.instrStr = instrStr
      self.portData = portData
      self.uops = uops
      self.retireSlots = retireSlots
      self.inputOperands = inputOperands # list registers
      self.indexesOfAddrRegs = indexesOfAddrRegs # indexes into inputOperands
      self.outputOperands = outputOperands
      self.latencies = latencies # latencies[(x,y)] = l; x,y are indices into input and output operands
      self.lcpStall = lcpStall
      self.modifiesStack = modifiesStack # pops or pushes to the stack

   def __repr__(self):
       return "Instr: " + str(self.__dict__)

class UnknownInstr(Instr):
   def __init__(self, asm, opcode, nPrefixes):
      Instr.__init__(self, asm, opcode, nPrefixes, instrStr='', portData={}, uops=0, retireSlots=1, inputOperands=[], indexesOfAddrRegs=[], outputOperands=[],
                     latencies={}, lcpStall=False, modifiesStack=False)

class RegOperand:
   def __init__(self, reg):
      self.reg = reg

class MemOperand:
   def __init__(self, memAddr):
      self.memAddr = memAddr

'''
class InstrInstance:
   def __init__(self, instr):
      self.instr = instr
      self.uops = []
      self.renamedInputOperands = []
      self.renamedOutputOperands = []
'''

class RenamedOperand:
   def __init__(self, ready=None):
      self.ready = ready


class FrontEnd:
   def __init__(self, uopGenerator, reorderBuffer, scheduler):
      self.uopGenerator = uopGenerator
      self.reorderBuffer = reorderBuffer
      self.scheduler = scheduler
      self.instructionQueue = deque()
      self.predecoder = PreDecoder(uopGenerator, self.instructionQueue)

   def cycle(self):
      self.predecoder.cycle()

      uops = []
      if len(self.reorderBuffer.uops) < RB_Width:
         while self.instructionQueue:
            instrUops = self.instructionQueue[0][1]
            if uops and (len(uops) + len(instrUops) > Decode_Width):
               break
            uops.extend(instrUops)
            self.instructionQueue.popleft()
            
         #uops = [uop for _ in xrange(min(FE_Width, len(self.instructionQueue))) for uop in self.instructionQueue.popleft()[1]]
      #print len(uops)
      IDQ = []
      for fusedUop in uops: # ToDo: unlamination
         if isinstance(fusedUop, PseudoFusedStackSynchUop):
            IDQ.extend(fusedUop.uops)
         else:
            IDQ.append(fusedUop)

      self.reorderBuffer.cycle(IDQ)
      self.scheduler.cycle([u for fusedUop in IDQ for u in fusedUop.getUnfusedUops() if u.possiblePorts])


class PreDecoder:
   def __init__(self, uopGenerator, instructionQueue):
      self.uopGenerator = uopGenerator
      self.instructionQueue = instructionQueue
      self.curOffset = 0 # can become negative if there are left-over instructions from the previous cycle
      self.curBlockQueue = deque()
      self.stalled = 0
      self.nextInstrUops = next(self.uopGenerator)

   def cycle(self):
      if not self.stalled:
         if not self.curBlockQueue: # ToDo: other microarch. than SKL
            while True:
               instr, uops = self.nextInstrUops
               instrLen = len(instr.opcode)/2               

               if self.curOffset + instrLen <= 16:
                  self.curBlockQueue.append((instr, uops))
                  self.curOffset += instrLen
                  self.nextInstrUops = next(self.uopGenerator)

                  if len(self.curBlockQueue) >= 5:
                     nextInstr, _ = self.nextInstrUops
                     if (self.curOffset + nextInstr.nPrefixes >= 16 or
                         (self.curOffset + nextInstr.nPrefixes == 15 and nextInstr.opcode[2*nextInstr.nPrefixes:].startswith('0F'))):
                        self.curOffset -= 16
                     break
               else:
                  self.curOffset -= 16
                  break
            
            if any(instr.lcpStall for instr, _ in self.curBlockQueue):
               self.stalled = sum(3 for instr, _ in self.curBlockQueue if instr.lcpStall)
            
         if not self.stalled and len(self.instructionQueue) < 25:
            self.instructionQueue.extend(self.curBlockQueue)
            self.curBlockQueue.clear()

      self.stalled = max(0, self.stalled-1)


class ReorderBuffer:
   def __init__(self, retireQueue):
      self.uops = deque()
      self.retireQueue = retireQueue

   def isEmpty(self):
      return not self.uops

   def cycle(self, newUops):
      self.retireUops()
      self.allocateUops(newUops)

   def retireUops(self):
      nRetiredInSameCycle = 0
      for _ in range(0, Retire_Width):
         if not self.uops: break
         fusedUop = self.uops[0]
         unfusedUops = fusedUop.getUnfusedUops()
         if all((u.executed is not None) for u in unfusedUops):
            self.uops.popleft()
            self.retireQueue.append(fusedUop)
            for u in unfusedUops:
               u.retired = clock
               u.retireIdx = nRetiredInSameCycle
            nRetiredInSameCycle += 1
         else:
            break

   def allocateUops(self, newUops):
      for fusedUop in newUops:
         self.uops.append(fusedUop)
         for uop in fusedUop.getUnfusedUops():
            uop.allocated = clock
            if not uop.possiblePorts:
               uop.executed = clock
               #for reg in uop.renamedOutputOperands:
               #   reg.ready = clock


class Scheduler:
   def __init__(self):
      self.portQueues = {p:deque() for p in allPorts}
      self.inflightUops = []

   def cycle(self, newUops):
      self.processInflightUops()
      self.dispatchUops()
      self.addNewUops(newUops)

   def dispatchUops(self):
      for port, queue in self.portQueues.items():
         for uop in queue:
            if not uop.renamedInputOperands or any((op.ready is not None) for op in uop.renamedInputOperands): # best-case assumption
               uop.dispatched = clock
               queue.remove(uop)
               self.inflightUops.append(uop)
               break

   def processInflightUops(self):
      for uop in self.inflightUops:
         if any((op.ready is None) for op in uop.renamedInputOperands):
            continue
         firstDispatchTime = min(uop2.dispatched for uop2 in uop.relatedUops)
         newReadyOutputOperands = set(oi for oi, op in enumerate(uop.renamedOutputOperands) if uop.renamedOutputOperands[oi].ready is None)
         for (ii,oi), l in uop.latencies.items():
            inpReady = uop.renamedInputOperands[ii].ready
            if (oi in newReadyOutputOperands) and inpReady >= 0 and max(firstDispatchTime, inpReady) + l > clock:
               newReadyOutputOperands.remove(oi)
         for oi in newReadyOutputOperands:
            uop.renamedOutputOperands[oi].ready = clock
         if all((op.ready is not None) for op in uop.renamedOutputOperands):
            uop.executed = clock
      self.inflightUops = [u for u in self.inflightUops if u.executed is None]

   def addNewUops(self, newUops):
      for uop in newUops:
         #minPortUsage = min(len(q) for p, q in self.portQueues.items() if p in uop.possiblePorts)
         #portCandidates = [p for p, q in self.portQueues.items() if (p in uop.possiblePorts) and (len(q) == minPortUsage)]
         port = min(((p,q) for p, q in self.portQueues.items() if p in uop.possiblePorts), key=lambda x: len(x[1]))[0] #random.choice(portCandidates)
         self.portQueues[port].append(uop)
         uop.actualPort = port


def getAllPorts():
   if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB']: return [str(i) for i in range(0,6)]
   elif arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL']: return [str(i) for i in range(0,8)]
   elif arch in ['ICL']: return [str(i) for i in range(0,10)]

MemAddr = namedtuple('MemAddr', ['base', 'index', 'scale', 'displacement'])
def getMemAddr(memAddrAsm):
   base = index = None
   displacement = 0
   scale = 1
   for c in re.split('\+|-', re.search('\[(.*)\]', memAddrAsm).group(1)):
      if '0x' in c:
         displacement = int(c, 0)
         if '-0x' in memAddrAsm:
            displacement = -displacement
      elif '*' in c:
         index, scale = c.split('*')
         scale = int(scale)
      else:
         base = c
   return MemAddr(base, index, scale, displacement)


# must only be called once for a given list of instructions
def adjustLatencies(instructions):
   prevWriteToReg = dict() # reg -> instr
   for instr in instructions:
      for outOp in instr.outputOperands:
         if isinstance(outOp, RegOperand):
            prevWriteToReg[getCanonicalReg(outOp.reg)] = instr
   for instr in instructions:
      for inOp in instr.inputOperands:
         if isinstance(inOp, MemOperand):
            memAddr = inOp.memAddr
            if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'KBL', 'CFL', 'SKX']:
               if (memAddr.index is None) and (memAddr.displacement < 2048):
                  if (memAddr.base in prevWriteToReg) and (prevWriteToReg[memAddr.base].instrStr in ['MOV (R64, M64)', 'MOV (RAX, M64)']):
                     for inI in instr.indexesOfAddrRegs:
                        for outI, _ in enumerate(instr.outputOperands):
                           instr.latencies[(inI, outI)] -= 1
      for outOp in instr.outputOperands:
         if isinstance(outOp, RegOperand):
            prevWriteToReg[getCanonicalReg(outOp.reg)] = instr


def getInstructions(filename, rawFile, iacaMarkers, instrDataDict):
   xedBinary = os.path.join(os.path.dirname(__file__), '..', 'XED-to-XML', 'obj', 'wkit', 'bin', 'xed')
   output = subprocess.check_output([xedBinary, '-64', '-v', '4', ('-ir' if rawFile else '-i'), filename])
   disas = parseXedOutput(output, iacaMarkers)

   instructions = []
   for instrD in disas:
      usedRegs = [getCanonicalReg(r) for _, r in instrD.regOperands.items() if r in GPRegs or 'MM' in r]
      sameReg = (len(usedRegs) > 1 and len(set(usedRegs)) == 1)
      nPrefixes = int(instrD.attributes.get('NPREFIXES', 0))
      lcpStall = ('PREFIX66' in instrD.attributes) and (instrD.attributes.get('IMM_WIDTH', '') == '16')
      modifiesStack = any(('STACK' in r) for r in instrD.regOperands.values())

      instruction = None
      for instrData in instrDataDict.get(instrD.iform, []):
         if all(instrD.attributes.get(k, '0') == v for k, v in instrData['attributes'].items()):
            uops = instrData.get('uops', 0)
            retireSlots = instrData.get('retireSlots', 0)
            latData = instrData.get('lat', dict())
            portData = instrData.get('ports', {})
            if sameReg:
               uops = instrData.get('uops_SameReg', uops)
               retireSlots = instrData.get('retireSlots_SameReg', retireSlots)
               latData = instrData.get('lat_SameReg', latData)
               portData = instrData.get('ports_SameReg', portData)

            instrInputRegOperands = [(n,r) for n, r in instrD.regOperands.items() if (not 'STACK' in r) and (('R' in instrD.rw[n]) or ('CW' in instrD.rw[n])
                                                                                                                             or (getRegSize(r) in [8, 16]))]
            instrInputMemOperands = [(n,m) for n, m in instrD.memOperands.items() if ('R' in instrD.rw[n]) or ('CW' in instrD.rw[n])]
            instrOutputRegOperands = [(n, r) for n, r in instrD.regOperands.items() if (not 'STACK' in r) and ('W' in instrD.rw[n])]
            instrOutputMemOperands = [(n, m) for n, m in instrD.memOperands.items() if 'W' in instrD.rw[n]]
            instrOutputOperands = instrOutputRegOperands + instrOutputMemOperands

            inputOperands = []
            outputOperands = [RegOperand(r) for _, r in instrOutputRegOperands] + [MemOperand(getMemAddr(m)) for _, m in instrOutputMemOperands]
            indexesOfAddrRegs = []

            latencies = dict()
            for inpN, inpR in instrInputRegOperands:
               inputOperands.append(RegOperand(inpR))
               for oi, (outN, _) in enumerate(instrOutputOperands):
                  latencies[(len(inputOperands)-1, oi)] = latData.get((inpN, outN), 1)

            for inpN, inpM in instrInputMemOperands:
               if 'AGEN' in inpN: continue
               memAddr = getMemAddr(inpM)
               inputOperands.append(MemOperand(memAddr))
               for oi, (outN, _) in enumerate(instrOutputOperands):
                  latencies[(len(inputOperands)-1, oi)] = latData.get((inpN, outN, 'mem'), 1)

            if not modifiesStack:
               for inpN, inpM in set(instrInputMemOperands + instrOutputMemOperands):
                  memAddr = getMemAddr(inpM)
                  for reg in [memAddr.base, memAddr.index]:
                     if reg is None: continue
                     inputOperands.append(RegOperand(reg))
                     indexesOfAddrRegs.append(len(inputOperands)-1)
                     for oi, (outN, _) in enumerate(instrOutputOperands):
                        if 'MM' in reg:
                           latencies[(len(inputOperands)-1, oi)] = latData.get((inpN, outN, 'addrVSIB'), 1)
                        else:
                           latencies[(len(inputOperands)-1, oi)] = latData.get((inpN, outN, 'addr'), 1)

            instruction = Instr(instrD.asm, instrD.opcode, nPrefixes, instrData['string'], portData, uops, retireSlots, inputOperands, indexesOfAddrRegs,
                                outputOperands, latencies, lcpStall, modifiesStack)
            break
      if instruction is None:
         instruction = UnknownInstr(instrD.asm, instrD.opcode, nPrefixes)
      instructions.append(instruction)
   return instructions


# repeatedly iterates over instructions; in each step, returns renamed uops for one instruction
def UopGenerator(instructions):
   stackPtrImplicitlyModified = False
   renameDict = {}
   for rnd in count():
      for instr in instructions:
         stackSynchUop = None
         if (stackPtrImplicitlyModified and
               any((getCanonicalReg(op.reg) == 'RSP') for op in instr.inputOperands+instr.outputOperands if isinstance(op, RegOperand))):
            renamedInputOperand = renameDict.get('RSP', RenamedOperand(-1))
            renamedOutputOperand = RenamedOperand()
            renameDict['RSP'] = renamedOutputOperand
            stackSynchUop = StackSynchUop(instr, rnd, renamedInputOperand, renamedOutputOperand)
            stackPtrImplicitlyModified = False

         if instr.modifiesStack:
            stackPtrImplicitlyModified = True
            
         renamedInputOperands = []
         renamedOutputOperands = []
         if instr.uops:
            for op in instr.inputOperands: # ToDo: partial register stalls
               if isinstance(op, RegOperand):
                  key = getCanonicalReg(op.reg)
               else:
                  key = tuple(renameDict.get(x, x) for x in op.memAddr)
               if not key in renameDict:
                  renameDict[key] = RenamedOperand(-1)
               renamedInputOperands.append(renameDict[key])
            for op in instr.outputOperands:
               if isinstance(op, RegOperand):
                  key = getCanonicalReg(op.reg)
               else:
                  key = tuple(renameDict.get(x, x) for x in op.memAddr)
               renamedOp = RenamedOperand()
               renameDict[key] = renamedOp
               renamedOutputOperands.append(renamedOp)
            
         else:
            if (len(instr.inputOperands) == 1 and len(instr.outputOperands) == 1 and isinstance(instr.inputOperands[0], RegOperand) and
                                                                                     isinstance(instr.outputOperands[0], RegOperand)):
               # Zero-latency mov
               canonicalInpReg = getCanonicalReg(instr.inputOperands[0].reg)
               canonicalOutReg = getCanonicalReg(instr.outputOperands[0].reg)
               if not canonicalInpReg in renameDict:
                  renameDict[canonicalInpReg] = RenamedOperand(-1)
               renameDict[canonicalOutReg] = renameDict[canonicalInpReg]
            else:
               for op in instr.outputOperands:
                  if isinstance(op, RegOperand):
                     key = getCanonicalReg(op.reg)
                  else:
                     key = tuple(renameDict.get(x, x) for x in op.memAddr)
                  renamedOp = RenamedOperand()
                  renamedOp.ready = -1
                  renameDict[key] = renamedOp

         unfusedUops = []

         remUops = instr.retireSlots
         for pc, n in instr.portData.items():
            remUops -= n
            for _ in range(0, n):
               ports = list(pc)
               unfusedUops.append(Uop(instr, rnd, ports, renamedInputOperands, renamedOutputOperands, dict(instr.latencies)))

         for _ in range(0, remUops):
            uop = Uop(instr, rnd, None, renamedInputOperands, renamedOutputOperands, dict(instr.latencies))
            unfusedUops.append(uop)

         for uop in unfusedUops:
            uop.relatedUops = list(unfusedUops)

         #uopsRoundDict[instr][ri].extend(unfusedUops)

         fusedDomainUops = []
         for _ in range(0, instr.uops - instr.retireSlots):
            if len(unfusedUops) < 2: break
            fusedDomainUops.append(FusedUop([unfusedUops.pop(), unfusedUops.pop()]))
         fusedDomainUops.extend(unfusedUops)

         if stackSynchUop is not None:
            fusedDomainUops[0] = PseudoFusedStackSynchUop([stackSynchUop, fusedDomainUops[0]])
         
         yield (instr, fusedDomainUops)
         #allUops.extend(fusedDomainUops)

def printPortUsage(instructions, uopsForRound):
   formatStr = '|' + '{:^9}|'*(len(allPorts)+1)

   print '-'*(1+10*(len(allPorts)+1))
   print formatStr.format('Uops', *allPorts)
   print '-'*(1+10*(len(allPorts)+1))
   portUsageC = Counter(uop.actualPort for uopsDict in uopsForRound for uops in uopsDict.values() for uop in uops)
   portUsageL = [('{:.2f}'.format(float(portUsageC[p])/len(uopsForRound)) if p in portUsageC else '') for p in allPorts]
   print formatStr.format(str(sum(len(uops) for uops in uopsForRound[0].values())), *portUsageL)
   print '-'*(1+10*(len(allPorts)+1))
   print ''

   print formatStr.format('Uops', *allPorts)
   print '-'*(1+10*(len(allPorts)+1))
   for instr in instructions:
      uopsForInstr = [uopsDict[instr] for uopsDict in uopsForRound]
      portUsageC = Counter(uop.actualPort for uops in uopsForInstr for uop in uops)
      portUsageL = [('{:.2f}'.format(float(portUsageC[p])/len(uopsForRound)) if p in portUsageC else '') for p in allPorts]

      uopsCol = str(len(uopsForInstr[0]))
      if isinstance(instr, UnknownInstr):
         uopsCol = 'X'

      print formatStr.format(uopsCol, *portUsageL) + ' ' + instr.asm


# Disassembles a binary and finds for each instruction the corresponding entry in the XML file.
# With the -iacaMarkers option, only the parts of the code that are between the IACA markers are considered.
def main():
   parser = argparse.ArgumentParser(description='Disassembler')
   parser.add_argument('filename', help="File to be disassembled")
   parser.add_argument("-iacaMarkers", help="Use IACA markers", action='store_true')
   parser.add_argument("-raw", help="raw file", action='store_true')
   parser.add_argument("-arch", help="Microarchitecture", default='SKL')
   args = parser.parse_args()

   global arch, allPorts
   arch = args.arch
   allPorts = getAllPorts()

   instrDataDict = importlib.import_module('instrData.'+arch).instrData

   instructions = getInstructions(args.filename, args.raw, args.iacaMarkers, instrDataDict)
   lastInstr = instructions[-1]
   adjustLatencies(instructions)
   #print instructions

   global clock
   clock = 0

   uopGenerator = UopGenerator(instructions)
   retireQueue = deque()
   rb = ReorderBuffer(retireQueue)
   scheduler = Scheduler()
   frontEnd = FrontEnd(uopGenerator, rb, scheduler)

   nRounds = 150
   uopsForRound = [{instr: [] for instr in instructions} for _ in range(0, nRounds)]

   done = False
   while True:
      frontEnd.cycle()
      while retireQueue:
         fusedUop = retireQueue.popleft()

         for uop in fusedUop.getUnfusedUops():
            instr = uop.instr
            rnd = uop.rnd
            if rnd >= nRounds:
               done = True
               break
            uopsForRound[rnd][instr].append(uop)
      if done:
         break

      clock += 1

   TP = None

   firstRelevantRound = 50
   lastRelevantRound = nRounds-1
   for rnd in range(lastRelevantRound, lastRelevantRound-5, -1):
      if uopsForRound[firstRelevantRound][lastInstr][-1].retireIdx == uopsForRound[rnd][lastInstr][-1].retireIdx:
         lastRelevantRound = rnd
         break

   uopsForRound = uopsForRound[firstRelevantRound:(lastRelevantRound+1)]

   TP = float(uopsForRound[-1][lastInstr][-1].retired - uopsForRound[0][lastInstr][-1].retired) / (len(uopsForRound)-1)
   #TP = max(float((uop2.retired-uop1.retired)) for d in uopsRoundDict.values() for (uop1, uop2) in zip(d[25], d[nRounds-25]))/(nRounds-50)

   print 'TP: {:.2f}'.format(TP)
   print ''

   printPortUsage(instructions, uopsForRound)


if __name__ == "__main__":
    main()

