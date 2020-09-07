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
Retire_Width = 4
RB_Width = 224
RS_Width = 97
#nDecoders = 5
Decode_Width = 5
IDQ_Width = 25
issue_Width = 4
IQ_Width = 25
issue_dispatch_delay = 5

class Uop:
   def __init__(self, instr, rnd, possiblePorts, inputOperands, outputOperands):
      self.instr = instr
      self.rnd = rnd # iteration round
      self.possiblePorts = possiblePorts
      self.inputOperands = inputOperands # [(instrInputOperand, renamedInpOperand), ...]
      self.outputOperands = outputOperands
      self.actualPort = None
      self.predecoded = None
      self.allocated = None
      self.issued = None
      self.dispatched = None
      self.executed = None
      self.retired = None
      self.retireIdx = None # how many other uops were already retired in the same cycle
      #self.relatedUops = [self]

   def getUnfusedUops(self):
      return [self]

   def __str__(self):
      return 'UopInstance(rnd: {}, p: {}, a: {}, d: {}, ex: {}, rt: {}, i: {}, o: {})'.format(self.rnd, self.actualPort, self.allocated, self.dispatched, self.executed, self.retired, self.inputOperands, self.outputOperands)


class FusedUop:
   def __init__(self, uops):
      self.uops = uops

   def getUnfusedUops(self):
      return self.uops


class LaminatedUop:
   def __init__(self, fusedUops):
      self.fusedUops = fusedUops

   def getFusedUops(self):
      return self.fusedUops


class StackSynchUop(Uop):
   def __init__(self, instr, rnd, renamedInputOperand, renamedOutputOperand):
      possiblePorts = (['0','1','5'] if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB'] else ['0','1','5','6'])
      Uop.__init__(self, instr, rnd, possiblePorts, [(None, renamedInputOperand)], [(None, renamedOutputOperand)])

# A combination of a StackSynchUop with a (possibly fused) uop that takes only one slot in the decoder; note: does not exist in the actual hardware
#class PseudoFusedStackSynchUop:
#   def __init__(self, uops):
#      self.uops = uops


class Instr:
   def __init__(self, asm, opcode, nPrefixes, instrStr, portData, uops, retireSlots, nLaminatedDomainUops, inputRegOperands, inputMemOperands,
                outputRegOperands, outputMemOperands, memAddrOperands, latencies, lcpStall, modifiesStack, complexDecoder, macroFusibleWith,
                macroFused=False):
      self.asm = asm
      self.opcode = opcode
      self.nPrefixes = nPrefixes
      self.instrStr = instrStr
      self.portData = portData
      self.uops = uops
      self.retireSlots = retireSlots
      self.nLaminatedDomainUops = nLaminatedDomainUops
      self.inputRegOperands = inputRegOperands
      self.inputMemOperands = inputMemOperands      
      self.outputRegOperands = outputRegOperands
      self.outputMemOperands = outputMemOperands
      self.memAddrOperands = memAddrOperands
      self.latencies = latencies # latencies[(inOp,outOp)] = l
      self.lcpStall = lcpStall
      self.modifiesStack = modifiesStack # pops or pushes to the stack
      self.complexDecoder = complexDecoder # requires the complex decoder
      self.macroFusibleWith = macroFusibleWith
      self.macroFused = macroFused # macro-fused with the previous instruction

   def __repr__(self):
       return "Instr: " + str(self.__dict__)

   def isZeroLatencyMovInstr(self):
      return ('MOV' in self.instrStr) and (not self.uops) and (len(self.inputRegOperands) == 1) and (len(self.outputRegOperands) == 1)


class UnknownInstr(Instr):
   def __init__(self, asm, opcode, nPrefixes):
      Instr.__init__(self, asm, opcode, nPrefixes, instrStr='', portData={}, uops=0, retireSlots=1, nLaminatedDomainUops=1, inputRegOperands=[],
                     inputMemOperands=[], outputRegOperands=[], outputMemOperands=[], memAddrOperands=[], latencies={}, lcpStall=False, modifiesStack=False,
                     complexDecoder=False, macroFusibleWith=set())

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
   def __init__(self, nonRenamedOperand=None):      
      self.nonRenamedOperand = nonRenamedOperand
      self.uops = [] # list of uops that need to have executed before this operand becomes ready
      self.__ready = None

   # Returns cycle in which operand became ready, or None if it is not yet ready
   def ready(self):      
      if (not self.uops) or (not any(uop.inputOperands for uop in self.uops)):
         return -1
      if any((uop.dispatched is None) for uop in self.uops):
         return None

      if self.__ready is None:
         self.__ready = -1
         for inpOp, renInpOp in set((op, rOp) for uop in self.uops for op, rOp in uop.inputOperands):
            minCycle = sys.maxsize
            for uop in self.uops:
               if not (inpOp, renInpOp) in uop.inputOperands: continue
               minCycle = min(minCycle, uop.dispatched + uop.instr.latencies.get((inpOp, self.nonRenamedOperand), 1))
            self.__ready = max(self.__ready, minCycle)
         
      if self.__ready <= clock:         
         return self.__ready
      return None


class FrontEnd:
   def __init__(self, uopGenerator, reorderBuffer, scheduler):
      self.uopGenerator = uopGenerator
      self.reorderBuffer = reorderBuffer
      self.scheduler = scheduler
      self.instructionQueue = deque()
      self.predecoder = PreDecoder(uopGenerator, self.instructionQueue)
      self.IDQ = deque()

   def cycle(self):      
      issueUops = []
      if not self.reorderBuffer.isFull() and not self.scheduler.isFull():
         while self.IDQ:
            fusedUops = self.IDQ[0].getFusedUops()
            if len(issueUops) + len(fusedUops) > issue_Width:
               break
            for fusedUop in fusedUops:
               for uop in fusedUop.getUnfusedUops():
                  uop.issued = clock
            issueUops.extend(fusedUops)
            self.IDQ.popleft()

      self.reorderBuffer.cycle(issueUops)
      self.scheduler.cycle(issueUops)

      laminatedDomainUops = []
      while self.instructionQueue:
         instr, instrUops = self.instructionQueue[0]
         if len(self.IDQ) + len(laminatedDomainUops) + len(instrUops) > IDQ_Width:
            break
         if laminatedDomainUops and (instr.complexDecoder or (len(laminatedDomainUops) + len(instrUops) > Decode_Width)):
            break
         for lamUop in instrUops:               
            for fusedUop in lamUop.getFusedUops():
               for uop in fusedUop.getUnfusedUops():
                  uop.allocated = clock
         laminatedDomainUops.extend(instrUops)
         self.instructionQueue.popleft()      
      self.IDQ.extend(laminatedDomainUops)

      self.predecoder.cycle()


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

         if not self.stalled and len(self.instructionQueue) < IQ_Width:
            for _, instrUops in self.curBlockQueue:
               for lamUop in instrUops:
                  for fusedUop in lamUop.getFusedUops():
                     for uop in fusedUop.getUnfusedUops():
                        uop.predecoded = clock
            self.instructionQueue.extend(self.curBlockQueue)
            self.curBlockQueue.clear()

      self.stalled = max(0, self.stalled-1)


class ReorderBuffer:
   def __init__(self, retireQueue):
      self.uops = deque()
      self.retireQueue = retireQueue

   def isEmpty(self):
      return not self.uops

   def isFull(self):
      return len(self.uops) + issue_Width > RB_Width

   def cycle(self, newUops):
      self.retireUops()
      self.addUops(newUops)

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

   def addUops(self, newUops):
      for fusedUop in newUops:
         self.uops.append(fusedUop)
         for uop in fusedUop.getUnfusedUops():
            if not uop.possiblePorts:
               uop.executed = clock


class Scheduler:
   def __init__(self):
      self.portQueues = {p:deque() for p in allPorts}
      self.portUsage = {p:0  for p in allPorts}
      self.uopsDispatchedInPrevCycle = [] # the port usage counter is decreased one cycle after uops are issued
      self.inflightUops = []

   def isFull(self):
      return sum(len(q) for q in self.portQueues.values()) + issue_Width > RS_Width

   def cycle(self, newUops):
      self.addNewUops(newUops)      
      self.processInflightUops()
      self.dispatchUops()      

   def dispatchUops(self):
      uopsDispatched = []
      for port, queue in self.portQueues.items():
         for uop in queue:
            if uop.issued + issue_dispatch_delay > clock:
               continue
            allOperandsReady = all((renamedOp.ready() is not None) for _, renamedOp in uop.inputOperands)
            if allOperandsReady: # worst-case assumption
               uop.dispatched = clock
               queue.remove(uop)
               uopsDispatched.append(uop)
               break
      self.inflightUops.extend(uopsDispatched)
      for uop in self.uopsDispatchedInPrevCycle:
         self.portUsage[uop.actualPort] -= 1
      self.uopsDispatchedInPrevCycle = uopsDispatched

   def processInflightUops(self):
      for uop in self.inflightUops:
         '''
         firstDispatchTime = min(uop2.dispatched for uop2 in uop.relatedUops)
         newReadyOutputOperands = set(oi for oi, op in enumerate(uop.renamedOutputOperands) if uop.renamedOutputOperands[oi].ready is None)
         for (ii,oi), l in uop.latencies.items():
            inpReady = uop.renamedInputOperands[ii].ready
            if (oi in newReadyOutputOperands) and inpReady >= 0 and max(firstDispatchTime, inpReady) + l > clock:
               newReadyOutputOperands.remove(oi)
         for oi in newReadyOutputOperands:
            uop.renamedOutputOperands[oi].ready = clock
         '''
         if all((renamedOp.ready() is not None) for _, renamedOp in uop.outputOperands):
            uop.executed = clock
      self.inflightUops = [u for u in self.inflightUops if u.executed is None]

   def addNewUops(self, newUops):
      #print len(newUops)
      prevPortUsage = dict(self.portUsage)
      for issueSlot, fusedUop in enumerate(newUops):
         for uop in fusedUop.getUnfusedUops():
            if not uop.possiblePorts:
               continue
            applicablePorts = [(p,u) for p, u in prevPortUsage.items() if p in uop.possiblePorts]
            minPort, minPortUsage = min(applicablePorts, key=lambda x: (x[1], -int(x[0]))) # port with minimum usage so far
            
            if issueSlot % 2 == 0 or len(applicablePorts) == 1:
               port = minPort
            else:
               remApplicablePorts = [(p, u) for p, u in applicablePorts if p != minPort]
               min2Port, min2PortUsage = min(remApplicablePorts, key=lambda x: (x[1], -int(x[0]))) # port with second smallest usage so far
               if min2PortUsage >= minPortUsage + 3:
                  port = minPort
               else:
                  port = min2Port

            #print str(issueSlot) + ': ' + str(port)
            self.portQueues[port].append(uop)
            self.portUsage[port] += 1
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
      for outOp in instr.outputRegOperands:
         if instr.isZeroLatencyMovInstr():
            prevWriteToReg[getCanonicalReg(outOp.reg)] = prevWriteToReg.get(getCanonicalReg(instr.inputRegOperands[0].reg), instr)
         else:
            prevWriteToReg[getCanonicalReg(outOp.reg)] = instr
   for instr in instructions:
      for inOp in instr.inputMemOperands:
         memAddr = inOp.memAddr
         if arch in ['SNB', 'IVB', 'HSW', 'BDW', 'SKL', 'KBL', 'CFL', 'SKX']:
            if (memAddr.base is not None) and (memAddr.index is None) and (0 <= memAddr.displacement < 2048):
               canonicalBaseReg = getCanonicalReg(memAddr.base)
               if (canonicalBaseReg in prevWriteToReg) and (prevWriteToReg[canonicalBaseReg].instrStr in ['MOV (R64, M64)', 'MOV (RAX, M64)',
                                                                                                          'MOV (R32, M32)', 'MOV (EAX, M32)']):
                  for memAddrOp in instr.memAddrOperands:
                     for outputOp in instr.outputRegOperands + instr.outputMemOperands:
                        instr.latencies[(memAddrOp, outputOp)] -= 1
      for outOp in instr.outputRegOperands:
         if instr.isZeroLatencyMovInstr():
            prevWriteToReg[getCanonicalReg(outOp.reg)] = prevWriteToReg.get(getCanonicalReg(instr.inputRegOperands[0].reg), instr)
         else:
            prevWriteToReg[getCanonicalReg(outOp.reg)] = instr


def getInstructions(filename, rawFile, iacaMarkers, instrDataDict):
   xedBinary = os.path.join(os.path.dirname(__file__), '..', 'XED-to-XML', 'obj', 'wkit', 'bin', 'xed')
   output = subprocess.check_output([xedBinary, '-64', '-v', '4', ('-ir' if rawFile else '-i'), filename])
   disas = parseXedOutput(output, iacaMarkers)

   instructions = []
   for instrD in disas:
      usedRegs = [getCanonicalReg(r) for _, r in instrD.regOperands.items() if r in GPRegs or 'MM' in r]
      sameReg = (len(usedRegs) > 1 and len(set(usedRegs)) == 1)
      usesIndexedAddr = any((getMemAddr(memOp).index is not None) for memOp in instrD.memOperands.values())
      nPrefixes = int(instrD.attributes.get('NPREFIXES', 0))
      lcpStall = ('PREFIX66' in instrD.attributes) and (instrD.attributes.get('IMM_WIDTH', '') == '16')
      modifiesStack = any(('STACK' in r) for r in instrD.regOperands.values())

      instruction = None
      for instrData in instrDataDict.get(instrD.iform, []):
         if all(instrD.attributes.get(k, '0') == v for k, v in instrData['attributes'].items()):
            uops = instrData.get('uops', 0)
            retireSlots = instrData.get('retSlots', 0)
            uopsMITE = instrData.get('uopsMITE', 0)
            uopsMS = instrData.get('uopsMS', 0)
            latData = instrData.get('lat', dict())
            portData = instrData.get('ports', {})
            complexDecoder = instrData.get('complDec', False)
            if sameReg:
               uops = instrData.get('uops_SR', uops)
               retireSlots = instrData.get('retSlots_SR', retireSlots)
               uopsMITE = instrData.get('uopsMITE_SR', uopsMITE)
               uopsMS = instrData.get('uopsMS_SR', uopsMS)
               latData = instrData.get('lat_SR', latData)
               portData = instrData.get('ports_SR', portData)
               complexDecoder = instrData.get('complDec_SR', complexDecoder)
            elif usesIndexedAddr:
               uops = instrData.get('uops_I', uops)
               retireSlots = instrData.get('retSlots_I', retireSlots)
               uopsMITE = instrData.get('uopsMITE_I', uopsMITE)
               uopsMS = instrData.get('uopsMS_I', uopsMS)
               portData = instrData.get('ports_I', portData)
               complexDecoder = instrData.get('complDec_I', complexDecoder)
            nLaminatedDomainUops = uopsMITE + uopsMS

            instrInputRegOperands = [(n,r) for n, r in instrD.regOperands.items() if (not 'IP' in r) and (not 'STACK' in r) and (('R' in instrD.rw[n]) or
                                                                                                        ('CW' in instrD.rw[n]) or (getRegSize(r) in [8, 16]))]
            instrInputMemOperands = [(n,m) for n, m in instrD.memOperands.items() if ('R' in instrD.rw[n]) or ('CW' in instrD.rw[n])]
            instrOutputRegOperands = [(n, r) for n, r in instrD.regOperands.items() if (not 'IP' in r) and (not 'STACK' in r) and ('W' in instrD.rw[n])]
            instrOutputMemOperands = [(n, m) for n, m in instrD.memOperands.items() if 'W' in instrD.rw[n]]
            instrOutputOperands = instrOutputRegOperands + instrOutputMemOperands

            inputRegOperands = []
            inputMemOperands = []            
            outputRegOperands = []
            outputMemOperands = []
            memAddrOperands = []
            
            outputOperandsDict = dict()
            for n, r in instrOutputRegOperands:
               regOp = RegOperand(r)
               outputRegOperands.append(regOp)
               outputOperandsDict[n] = regOp
            for n, m in instrOutputMemOperands:
               memOp = MemOperand(getMemAddr(m))
               outputMemOperands.append(memOp)
               outputOperandsDict[n] = memOp
               
            latencies = dict()
            for inpN, inpR in instrInputRegOperands:
               regOp = RegOperand(inpR)
               inputRegOperands.append(regOp)
               for outN, _ in instrOutputOperands:
                  latencies[(regOp, outputOperandsDict[outN])] = latData.get((inpN, outN), 1)

            for inpN, inpM in instrInputMemOperands:
               if 'AGEN' in inpN: continue
               memOp = MemOperand(getMemAddr(inpM))
               inputMemOperands.append(memOp)
               for outN, _ in instrOutputOperands:
                  latencies[(memOp, outputOperandsDict[outN])] = latData.get((inpN, outN, 'mem'), 1)

            if not modifiesStack:
               for inpN, inpM in set(instrInputMemOperands + instrOutputMemOperands):
                  memAddr = getMemAddr(inpM)
                  for reg, addrType in [(memAddr.base, 'addr'), (memAddr.index, 'addrI')]:
                     if (reg is None) or ('IP' in reg): continue
                     regOp = RegOperand(reg)
                     memAddrOperands.append(regOp)
                     for outN, _ in instrOutputOperands:
                        latencies[(regOp, outputOperandsDict[outN])] = latData.get((inpN, outN, addrType), 1)

            instruction = Instr(instrD.asm, instrD.opcode, nPrefixes, instrData['string'], portData, uops, retireSlots, nLaminatedDomainUops, inputRegOperands,
                                inputMemOperands, outputRegOperands, outputMemOperands, memAddrOperands, latencies, lcpStall, modifiesStack,
                                complexDecoder, instrData.get('macroFusible', set()))
            #print instruction
            break

      if instruction is None:
         instruction = UnknownInstr(instrD.asm, instrD.opcode, nPrefixes)

      # Macro-fusion
      if instructions:
         prevInstr = instructions[-1]
         if instruction.instrStr in prevInstr.macroFusibleWith:
            instruction.macroFused = True
            instrPorts = instruction.portData.keys()[0]
            if prevInstr.uops == 0:
               prevInstr.uops = instruction.uops
               prevInstr.portData = instruction.portData
            else:
               for p, u in prevInstr.portData.items():
                  if set(instrPorts).issubset(set(p)):
                     del prevInstr.portData[p]
                     prevInstr.portData[instrPorts] = u

      instructions.append(instruction)
   return instructions


# repeatedly iterates over instructions; in each step, returns renamed uops for one instruction
def UopGenerator(instructions):
   stackPtrImplicitlyModified = False
   renameDict = {}
   for rnd in count():
      for instr in instructions:
         if instr.macroFused: continue

         allRegOperands = instr.inputRegOperands + instr.outputRegOperands + instr.memAddrOperands
      
         stackSynchUop = None
         if stackPtrImplicitlyModified and any((getCanonicalReg(op.reg) == 'RSP') for op in allRegOperands):
            renamedInputOperand = renameDict.get('RSP', RenamedOperand())
            renamedOutputOperand = RenamedOperand()
            renameDict['RSP'] = renamedOutputOperand
            stackSynchUop = StackSynchUop(instr, rnd, renamedInputOperand, renamedOutputOperand)
            renamedOutputOperand.uops = [stackSynchUop]
            stackPtrImplicitlyModified = False

         if instr.modifiesStack:
            stackPtrImplicitlyModified = True

         renamedInputRegOperands = []         
         renamedOutputRegOperands = []
         renamedInputMemOperands = []
         renamedOutputMemOperands = []
         renamedMemAddrOperands = []
         
         if instr.uops:
            for op in instr.inputRegOperands: # ToDo: partial register stalls
               key = getCanonicalReg(op.reg)
               if not key in renameDict:
                  renameDict[key] = RenamedOperand(op)
               renamedInputRegOperands.append((op, renameDict[key]))
            for op in instr.memAddrOperands:
               key = getCanonicalReg(op.reg)
               if not key in renameDict:
                  renameDict[key] = RenamedOperand(op)
               renamedMemAddrOperands.append((op, renameDict[key]))
            for op in instr.inputMemOperands:
               key = tuple(renameDict.get(x, x) for x in op.memAddr)
               if not key in renameDict:
                  renameDict[key] = RenamedOperand(op)
               renamedInputMemOperands.append((op, renameDict[key]))
            for op in instr.outputRegOperands:
               renamedOp = RenamedOperand(op)
               renameDict[getCanonicalReg(op.reg)] = renamedOp
               renamedOutputRegOperands.append((op, renamedOp))
            for op in instr.outputMemOperands:
               key = tuple(renameDict.get(x, x) for x in op.memAddr)
               renamedOp = RenamedOperand(op)
               renameDict[key] = renamedOp
               renamedOutputMemOperands.append((op, renamedOp))
         else:
            if instr.isZeroLatencyMovInstr():
               canonicalInpReg = getCanonicalReg(instr.inputRegOperands[0].reg)
               canonicalOutReg = getCanonicalReg(instr.outputRegOperands[0].reg)
               if not canonicalInpReg in renameDict:
                  renameDict[canonicalInpReg] = RenamedOperand()
               renameDict[canonicalOutReg] = renameDict[canonicalInpReg]
            else:
               for op in instr.outputRegOperands:
                  renameDict[getCanonicalReg(op.reg)] = RenamedOperand()
                  
         unfusedUops = []
         remUops = instr.retireSlots
         for pc, n in instr.portData.items():
            remUops -= n
            for _ in range(0, n):
               ports = list(pc)
               if any((p in ports) for p in ['2', '3', '7', '8']):
                  applicableInputOperands = renamedMemAddrOperands + renamedInputMemOperands # ToDo: distinguish between load and store operations
                  applicableOutputOperands = renamedOutputRegOperands + renamedOutputMemOperands
               elif any((p in ports) for p in ['4', '9']):
                  applicableInputOperands = renamedInputRegOperands + renamedMemAddrOperands + renamedInputMemOperands
                  applicableOutputOperands = renamedOutputMemOperands
               else:
                  applicableInputOperands = renamedInputRegOperands + renamedMemAddrOperands + renamedInputMemOperands
                  applicableOutputOperands = renamedOutputRegOperands + renamedOutputMemOperands
               uop = Uop(instr, rnd, ports, applicableInputOperands, applicableOutputOperands)
               unfusedUops.append(uop)
               for _, renamedOutputOp in applicableOutputOperands:
                  renamedOutputOp.uops.append(uop)
                  
         for _ in range(0, remUops):
            uop = Uop(instr, rnd, None, [], [])
            unfusedUops.append(uop)

         fusedDomainUops = []
         for _ in range(0, instr.retireSlots-1):
            fusedDomainUops.append(FusedUop([unfusedUops.pop()]))
         fusedDomainUops.append(FusedUop(unfusedUops))
         if stackSynchUop is not None:
            fusedDomainUops.append(stackSynchUop)

         laminatedDomainUops = []
         for _ in range(0, instr.nLaminatedDomainUops-1):
            laminatedDomainUops.append(LaminatedUop([fusedDomainUops.pop()]))
         laminatedDomainUops.append(LaminatedUop(fusedDomainUops))

         #if stackSynchUop is not None:
         #   fusedDomainUops[0] = PseudoFusedStackSynchUop([stackSynchUop, fusedDomainUops[0]])

         yield (instr, laminatedDomainUops)
         #allUops.extend(fusedDomainUops)

def printPortUsage(instructions, uopsForRound):
   formatStr = '|' + '{:^9}|'*(len(allPorts)+1)

   print '-'*(1+10*(len(allPorts)+1))
   print formatStr.format('Uops', *allPorts)
   print '-'*(1+10*(len(allPorts)+1))
   portUsageC = Counter(uop.actualPort for uopsDict in uopsForRound for uops in uopsDict.values() for uop in uops)
   portUsageL = [('{:.2f}'.format(float(portUsageC[p])/len(uopsForRound)) if p in portUsageC else '') for p in allPorts]
   #print formatStr.format(str(sum(len(uops) for uops in uopsForRound[0].values())), *portUsageL)
   print formatStr.format(str(sum(instr.uops for instr in instructions if not instr.macroFused)), *portUsageL)
   print '-'*(1+10*(len(allPorts)+1))
   print ''

   print formatStr.format('Uops', *allPorts)
   print '-'*(1+10*(len(allPorts)+1))
   for instr in instructions:
      uopsForInstr = [uopsDict[instr] for uopsDict in uopsForRound]
      portUsageC = Counter(uop.actualPort for uops in uopsForInstr for uop in uops)
      portUsageL = [('{:.2f}'.format(float(portUsageC[p])/len(uopsForRound)) if p in portUsageC else '') for p in allPorts]

      uopsCol = str(instr.uops)
      if isinstance(instr, UnknownInstr):
         uopsCol = 'X'
      elif instr.macroFused:
         uopsCol = 'M'

      print formatStr.format(uopsCol, *portUsageL) + ' ' + instr.asm


def writeHtmlFile(filename, title, head, body):
   with open(filename, "w") as f:
      f.write('<html>\n'
              '<head>\n'
              '<title>' + title + '</title>\n'
              + head +
              '</head>\n'
              '<body>\n'
              + body +
              '</body>\n'
              '</html>\n')


def generateHTMLTraceTable(filename, instructions, uopsForRound, maxCycle):
   style = []
   style.append('<style>')
   style.append('table {border-collapse: collapse}')   
   style.append('table, td, th {border: 1px solid black}')
   style.append('th {text-align: left; padding: 6px}')
   style.append('td {text-align: center}')
   style.append('code {white-space: nowrap}')   
   style.append('</style>')
   table = []
   table.append('<table>')
   table.append('<tr>')
   table.append('<th rowspan="2">It.</th>')
   table.append('<th rowspan="2">Instruction</th>')
   table.append('<th colspan="2" style="text-align:center">&mu;ops</th>')
   table.append('<th rowspan="2" colspan="{}">Cycles</th>'.format(maxCycle+1))
   table.append('</tr>')
   table.append('<tr>')
   table.append('<th style="text-align:center">Possible Ports</th>')
   table.append('<th style="text-align:center">Actual Port</th>')   
   table.append('</tr>')

   nUops = sum(len(uops) for uops in uopsForRound[0].values())
   for rnd, uopsDict in enumerate(uopsForRound):
      table.append('<tr style="border-top: 2px solid black">')
      table.append('<td rowspan="{}">{}</td>'.format(nUops, rnd))
      for instrI, instr in enumerate(instructions):
         if instrI > 0:
            table.append('<tr>')
         table.append('<td rowspan=\"{}\" style="text-align:left"><code>{}</code></td>'.format(len(uopsDict[instr]), instr.asm))
         for uopI, uop in enumerate(uopsDict[instr]):
            if uopI > 0:
               table.append('<tr>')
            table.append('<td>{{{}}}</td>'.format(','.join(uop.possiblePorts) if uop.possiblePorts else '-'))
            table.append('<td>{}</td>'.format(uop.actualPort if uop.actualPort else '-'))

            uopEvents = ['' for _ in range(0,maxCycle+1)]
            for evCycle, ev in [(uop.predecoded, 'P'), (uop.allocated, 'A'), (uop.issued, 'I'), (uop.dispatched, 'D'), (uop.executed, 'E'), (uop.retired, 'R')]:
               if evCycle is not None and evCycle <= maxCycle:
                  uopEvents[evCycle] += ev
                  
            for ev in uopEvents:
               table.append('<td>{}</td>'.format(ev))
            
            table.append('</tr>')
      

   table.append('</table>')
   writeHtmlFile(filename, 'Trace', '\n'.join(style), '\n'.join(table))
   

# Disassembles a binary and finds for each instruction the corresponding entry in the XML file.
# With the -iacaMarkers option, only the parts of the code that are between the IACA markers are considered.
def main():
   parser = argparse.ArgumentParser(description='Disassembler')
   parser.add_argument('filename', help="File to be disassembled")
   parser.add_argument("-iacaMarkers", help="Use IACA markers", action='store_true')
   parser.add_argument("-raw", help="raw file", action='store_true')
   parser.add_argument("-arch", help="Microarchitecture", default='CFL')
   parser.add_argument("-trace", help="HTML trace", nargs='?', const='trace.html')
   args = parser.parse_args()

   global arch, allPorts
   arch = args.arch
   allPorts = getAllPorts()

   instrDataDict = importlib.import_module('instrData.'+arch).instrData

   instructions = getInstructions(args.filename, args.raw, args.iacaMarkers, instrDataDict)
   lastApplicableInstr = [instr for instr in instructions if not instr.macroFused][-1] # ignore macro-fused instr.
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
   uopsForRound = []
   
   
   done = False
   while True:
      frontEnd.cycle()
      while retireQueue:
         fusedUop = retireQueue.popleft()

         for uop in fusedUop.getUnfusedUops():
            instr = uop.instr
            rnd = uop.rnd
            if rnd >= nRounds and clock > 500:
               done = True
               break
            if rnd >= len(uopsForRound):
               uopsForRound.append({instr: [] for instr in instructions})
            uopsForRound[rnd][instr].append(uop)
      
      if done:
         break

      clock += 1      

   TP = None

   firstRelevantRound = 50
   lastRelevantRound = len(uopsForRound)-2 # last round may be incomplete, thus -2   
   for rnd in range(lastRelevantRound, lastRelevantRound-5, -1):
      if uopsForRound[firstRelevantRound][lastApplicableInstr][-1].retireIdx == uopsForRound[rnd][lastApplicableInstr][-1].retireIdx:
         lastRelevantRound = rnd
         break

   uopsForRelRound = uopsForRound[firstRelevantRound:(lastRelevantRound+1)]

   TP = float(uopsForRelRound[-1][lastApplicableInstr][-1].retired - uopsForRelRound[0][lastApplicableInstr][-1].retired) / (len(uopsForRelRound)-1)
   #TP = max(float((uop2.retired-uop1.retired)) for d in uopsRoundDict.values() for (uop1, uop2) in zip(d[25], d[nRounds-25]))/(nRounds-50)

   print 'TP: {:.2f}'.format(TP)
   print ''

   printPortUsage(instructions, uopsForRelRound)

   if args.trace is not None:
      generateHTMLTraceTable(args.trace, instructions, uopsForRound, clock-1)

if __name__ == "__main__":
    main()

