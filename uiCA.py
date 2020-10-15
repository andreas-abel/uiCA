#!/usr/bin/python


import importlib
import os
import random
import re
import sys
from collections import Counter, defaultdict, deque, namedtuple
from heapq import heappop, heappush
from itertools import chain, count
from x64_lib import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../XED-to-XML'))
from disas import *

arch = None
clock = 0
allPorts = []
Retire_Width = 4
RB_Width = 224
RS_Width = 97
PreDecode_Width = 5
nDecoders = 4 #wikichip seems to be wrong
MITE_Width = 5
DSB_Width = 6
IDQ_Width = 25
issue_Width = 4
IQ_Width = 25
issue_dispatch_delay = 5

class Uop:
   idx_iter = count()

   def __init__(self, instr, rnd, possiblePorts, inputOperands, outputOperands, divCycles=0):
      self.idx = next(self.idx_iter)
      self.instr = instr
      self.rnd = rnd # iteration round
      self.possiblePorts = possiblePorts
      self.inputOperands = inputOperands
      self.outputOperands = outputOperands
      self.divCycles = divCycles
      self.renamedInputOperands = [] # [op[1] for op in inputOperands] # [(instrInputOperand, renamedInpOperand), ...]
      self.renamedOutputOperands = [] # [op[1] for op in outputOperands]
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
      return 'UopInstance(idx: {}, rnd: {}, p: {}, a: {}, d: {}, ex: {}, rt: {}, i: {}, o: {})'.format(self.idx, self.rnd, self.actualPort, self.allocated, self.dispatched, self.executed, self.retired, self.inputOperands, self.outputOperands)


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

   def getUnfusedUops(self):
      return [uop for fusedUop in self.getFusedUops() for uop in fusedUop.getUnfusedUops()]


class StackSynchUop(Uop):
   def __init__(self, instr, rnd):
      possiblePorts = (['0','1','5'] if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB'] else ['0','1','5','6'])
      Uop.__init__(self, instr, rnd, possiblePorts, [RegOperand('RSP')], [RegOperand('RSP')])

# A combination of a StackSynchUop with a (possibly fused) uop that takes only one slot in the decoder; note: does not exist in the actual hardware
#class PseudoFusedStackSynchUop:
#   def __init__(self, uops):
#      self.uops = uops


class Instr:
   def __init__(self, asm, opcode, nPrefixes, instrStr, portData, uops, retireSlots, nLaminatedDomainUops, divCycles, inputRegOperands, inputMemOperands,
                outputRegOperands, outputMemOperands, memAddrOperands, latencies, lcpStall, modifiesStack, complexDecoder, mayBeEliminated, isBranchInstr,
                macroFusibleWith, macroFusedWithPrevInstr=False, macroFusedWithNextInstr=False):
      self.asm = asm
      self.opcode = opcode
      self.nPrefixes = nPrefixes
      self.instrStr = instrStr
      self.portData = portData
      self.uops = uops
      self.retireSlots = retireSlots
      self.nLaminatedDomainUops = nLaminatedDomainUops
      self.divCycles = divCycles
      self.inputRegOperands = inputRegOperands
      self.inputMemOperands = inputMemOperands
      self.outputRegOperands = outputRegOperands
      self.outputMemOperands = outputMemOperands
      self.memAddrOperands = memAddrOperands
      self.latencies = latencies # latencies[(inOp,outOp)] = l
      self.lcpStall = lcpStall
      self.modifiesStack = modifiesStack # pops or pushes to the stack
      self.complexDecoder = complexDecoder # requires the complex decoder
      self.mayBeEliminated = mayBeEliminated # a move instruction that may be eliminated
      self.isBranchInstr = isBranchInstr
      self.macroFusibleWith = macroFusibleWith
      self.macroFusedWithPrevInstr = macroFusedWithPrevInstr
      self.macroFusedWithNextInstr = macroFusedWithNextInstr

   def __repr__(self):
       return "Instr: " + str(self.__dict__)

class UnknownInstr(Instr):
   def __init__(self, asm, opcode, nPrefixes):
      Instr.__init__(self, asm, opcode, nPrefixes, instrStr='', portData={}, uops=0, retireSlots=1, nLaminatedDomainUops=1, divCycles=0, inputRegOperands=[],
                     inputMemOperands=[], outputRegOperands=[], outputMemOperands=[], memAddrOperands=[], latencies={}, lcpStall=False, modifiesStack=False,
                     mayBeEliminated=False, complexDecoder=False, isBranchInstr=False, macroFusibleWith=set())


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


class Renamer:
   def __init__(self):
      self.renameDict = {}

      # renamed operands written by current instr.; this is necessary because we we generally don't know which uop of an instruction writes an operand
      self.curInstrRndRenameDict = {}
      self.curInstrRnd = (None, None, None)

      self.renGPRUsedForMoveElim = set()
      self.renGPRUsedForMoveElimInPrevCycle = set()
      self.nEliminationsInPrevCycle = 0

   def cycle(self, uops):
      nMoveEliminationsPossible = 4 #- max(self.nEliminationsInPrevCycle, len(self.renGPRUsedForMoveElimInPrevCycle)) # ToDo: other microarchitectures; Ice Lake seems to be unlimited
      #print self.renGPRUsedForMoveElimInPrevCycle
      #print str(nMoveEliminationsPossible) + ' ' + str(len(uops))
      self.nEliminationsInPrevCycle = 0
      #self.renGPRUsedForMoveElimInPrevCycle = set(self.renGPRUsedForMoveElim)

      for fusedUop in uops:
         for uop in fusedUop.getUnfusedUops():
            if (uop.instr, uop.rnd, isinstance(uop, StackSynchUop)) != self.curInstrRnd:
               self.curInstrRnd = (uop.instr, uop.rnd, isinstance(uop, StackSynchUop))
               self.renameDict.update(self.curInstrRndRenameDict)
               self.curInstrRndRenameDict.clear()

            if uop.instr.mayBeEliminated and (nMoveEliminationsPossible > 0) and not isinstance(uop, StackSynchUop):
               uop.possiblePorts = []
               canonicalInpReg = getCanonicalReg(uop.instr.inputRegOperands[0].reg)
               canonicalOutReg = getCanonicalReg(uop.instr.outputRegOperands[0].reg)
               renamedReg = self.renameDict.setdefault(canonicalInpReg, RenamedOperand())
               self.curInstrRndRenameDict[canonicalOutReg] = renamedReg
               self.renGPRUsedForMoveElim.add(renamedReg) # ToDo: non-GPR
               nMoveEliminationsPossible -= 1
               self.nEliminationsInPrevCycle += 1
            elif uop.instr.uops or isinstance(uop, StackSynchUop):
               for inpOp in uop.inputOperands:
                  if isinstance(inpOp, RegOperand):
                     key = getCanonicalReg(inpOp.reg) # ToDo: partial register stalls
                  else:
                     key = tuple(self.renameDict.get(x, x) for x in inpOp.memAddr)
                  renamedOp = self.renameDict.setdefault(key, RenamedOperand(inpOp))
                  uop.renamedInputOperands.append(renamedOp)
               for outOp in uop.outputOperands:
                  if isinstance(outOp, RegOperand):
                     key = getCanonicalReg(outOp.reg)
                  else:
                     key = tuple(self.renameDict.get(x, x) for x in outOp.memAddr)
                  renamedOp = self.curInstrRndRenameDict.setdefault(key, RenamedOperand(outOp))
                  uop.renamedOutputOperands.append(renamedOp)
                  renamedOp.uops.append(uop)
            else:
               # e.g., xor rax, rax
               for op in uop.instr.outputRegOperands:
                  self.curInstrRndRenameDict[getCanonicalReg(op.reg)] = RenamedOperand()

      #self.renGPRUsedForMoveElim &= set(self.renameDict.values())

      return uops


class FrontEnd:
   def __init__(self, instructions, reorderBuffer, scheduler, unroll):
      self.IDQ = deque()
      self.renamer = Renamer()
      self.reorderBuffer = reorderBuffer
      self.scheduler = scheduler
      self.unroll = unroll

      instructionQueue = deque()
      self.preDecoder = PreDecoder(instructionQueue)
      self.decoder = Decoder(instructionQueue)

      self.DSB = DSB()
      self.addressesInDSB = set()

      self.uopSource = 'MITE'
      if unroll:
         self.cacheBlockGenerator = CacheBlockGenerator(instructions, True)
      else:
         self.cacheBlocksForNextRoundGenerator = CacheBlocksForNextRoundGenerator(instructions)
         cacheBlocksForFirstRound = next(self.cacheBlocksForNextRoundGenerator)
         self.findCacheableAddresses(cacheBlocksForFirstRound)
         for cacheBlock in cacheBlocksForFirstRound:
            self.addNewCacheBlock(cacheBlock)
         if 0 in self.addressesInDSB:
            self.uopSource = 'DSB'

   def cycle(self):
      renamerUops = []
      if len(self.IDQ) >= issue_Width and not self.reorderBuffer.isFull() and not self.scheduler.isFull():
         while self.IDQ:
            fusedUops = self.IDQ[0].getFusedUops()
            if len(renamerUops) + len(fusedUops) > issue_Width:
               break
            renamerUops.extend(fusedUops)
            self.IDQ.popleft()

      issueUops = self.renamer.cycle(renamerUops)
      for fusedUop in issueUops:
         for uop in fusedUop.getUnfusedUops():
            uop.issued = clock

      self.reorderBuffer.cycle(issueUops)
      self.scheduler.cycle(issueUops)

      if len(self.IDQ) + DSB_Width > IDQ_Width:
         return

      # add new cache blocks
      while len(self.DSB.B32BlockQueue) < 2 and len(self.preDecoder.B16BlockQueue) < 4:
         if self.unroll:
            self.addNewCacheBlock(next(self.cacheBlockGenerator))
         else:
            for cacheBlock in next(self.cacheBlocksForNextRoundGenerator):
               self.addNewCacheBlock(cacheBlock)

      # add new uops to IDQ
      if self.uopSource == 'MITE':
         newInstrIUops = self.decoder.cycle()
         self.IDQ.extend(instrIuop[1] for instrIuop in newInstrIUops)
         if not self.unroll and newInstrIUops:
            curInstrI = newInstrIUops[-1][0]
            if curInstrI.instr.isBranchInstr or curInstrI.instr.macroFusedWithNextInstr:
               if 0 in self.addressesInDSB:
                  self.uopSource = 'DSB'
         self.preDecoder.cycle()
      elif self.uopSource == 'DSB':
         newInstrIUops = self.DSB.cycle()
         self.IDQ.extend(instrIuop[1] for instrIuop in newInstrIUops)
         if newInstrIUops and not self.DSB.isBusy():
            curInstrI = newInstrIUops[-1][0]
            if curInstrI.instr.isBranchInstr or curInstrI.instr.macroFusedWithNextInstr:
               nextAddr = 0
            else:
               nextAddr = curInstrI.address + len(curInstrI.instr.opcode)/2
            if nextAddr not in self.addressesInDSB:
               self.uopSource = 'MITE'

   def findCacheableAddresses(self, cacheBlocksForOneRound):
      for cacheBlock in cacheBlocksForOneRound:
         B32Blocks = [block for block in split64ByteBlockTo32ByteBlocks(cacheBlock) if block]
         if all(self.canBeCached(block) for block in B32Blocks):
            # on SKL, a 64-Byte block cannot be cached if the first or the second 32 Bytes cannot be cached
            # ToDo: other microarchitectures
            for B32Block in B32Blocks:
               self.addressesInDSB.add(B32Block[0].address)

   def canBeCached(self, B32Block):
      if sum(len(instrI.uops) for instrI in B32Block if not instrI.instr.macroFusedWithPrevInstr) > 18:
         # a 32-Byte block cannot be cached if it contains more than 18 uops
         return False
      lastInstrI = B32Block[-1]
      if lastInstrI.instr.macroFusedWithNextInstr:
         return False
      if lastInstrI.instr.isBranchInstr and (lastInstrI.address % 32) + len(lastInstrI.instr.opcode)/2 >= 32:
         # on SKL, if the next instr. after a branch starts in a new block, the current block cannot be cached
         # ToDo: other microarchitectures
         return False
      return True

   def addNewCacheBlock(self, cacheBlock):
      B32Blocks = split64ByteBlockTo32ByteBlocks(cacheBlock)
      for B32Block in B32Blocks:
         if not B32Block: continue
         if B32Block[0].address in self.addressesInDSB:
            d = deque(instrI for instrI in B32Block if not instrI.instr.macroFusedWithPrevInstr)
            if d:
               self.DSB.B32BlockQueue.append(d)
         else:
            for B16Block in split32ByteBlockTo16ByteBlocks(B32Block):
               if not B16Block: continue
               self.preDecoder.B16BlockQueue.append(deque(B16Block))
               lastInstrI = B16Block[-1]
               if lastInstrI.instr.isBranchInstr and (lastInstrI.address % 16) + len(lastInstrI.instr.opcode)/2 > 16:
                  # branch instr. ends in next block
                  self.preDecoder.B16BlockQueue.append(deque())



class DSB:
   def __init__(self):
      self.B32BlockQueue = deque()
      self.busy = False

   def cycle(self):
      self.busy = True
      B32Block = self.B32BlockQueue[0]

      retList = []
      while B32Block and (len(retList) < DSB_Width):
         self.addUopsToList(B32Block, retList)

      if not B32Block:
         self.B32BlockQueue.popleft()
         self.busy = False

         if self.B32BlockQueue and (len(retList) < DSB_Width):
            prevInstrI = retList[-1][0]
            if prevInstrI.address + len(prevInstrI.instr.opcode)/2 == self.B32BlockQueue[0][0].address: # or prevInstrI.instr.isBranchInstr or prevInstrI.instr.macroFusedWithNextInstr:
               self.busy = True
               B32Block = self.B32BlockQueue[0]
               while B32Block and (len(retList) < DSB_Width):
                  self.addUopsToList(B32Block, retList)

               if not B32Block:
                  self.B32BlockQueue.popleft()
                  self.busy = False

      return retList

   def addUopsToList(self, B32Block, lst):
      while B32Block and (len(lst) < DSB_Width):
         instrI = B32Block.popleft()
         lamUops = instrI.uops
         for lamUop in lamUops:
            lst.append((instrI, lamUop))
            for uop in lamUop.getUnfusedUops():
               uop.allocated = clock

   def isBusy(self):
      return self.busy

   '''
   def addUopsForNextRound(self):
      cacheBlocks = next(self.cacheBlocksForNextRoundGenerator)
      addToDecoder = False
      for cacheBlock in cacheBlocks:
         if not addToDecoder:
            B32Blocks = split64ByteBlockTo32ByteBlocks(cacheBlock)
            if any(sum(len(instrI.uops) for instrI in block) > 18 for block in B32Blocks):
               addToDecoder = True
            if cacheBlock == cacheBlocks[-1]:
               lastInstrI = cacheBlock[-1]
               if lastInstrI.instr.macroFusedWithPrevInstr:
                  addr = (cacheBlock[-2].address if len(cacheBlock) > 1 else cacheBlocks[-2][-1].address)
               else:
                  addr = lastInstrI.address
               nextInstrAddr = lastInstrI.address + len(lastInstrI.instr.opcode)/2
               if addr / 32 != nextInstrAddr / 32:
                  addToDecoder = True
         if not addToDecoder:
            self.buffer.extend(uop for instrI in cacheBlock for uop in instrI.uops)
         else:
            for B16Block in split64ByteBlockTo16ByteBlocks(cacheBlock):
               self.preDecoder.B16BlockQueue.append(deque(B16Block))
   '''


class Decoder:
   def __init__(self, instructionQueue):
      self.instructionQueue = instructionQueue

   def cycle(self):
      retList = []
      nDecodedInstrs = 0
      while self.instructionQueue:
         instrI = self.instructionQueue[0]
         if retList and (instrI.instr.complexDecoder or (len(retList) + len(instrI.uops) > MITE_Width)):
            break
         for lamUop in instrI.uops:
            retList.append((instrI, lamUop))
            for fusedUop in lamUop.getFusedUops():
               for uop in fusedUop.getUnfusedUops():
                  uop.allocated = clock
         self.instructionQueue.popleft()
         nDecodedInstrs += 1
         if nDecodedInstrs >= nDecoders:
            break
         if instrI.instr.isBranchInstr or instrI.instr.macroFusedWithNextInstr:
            break
      return retList

   def isEmpty(self):
      return (not self.instructionQueue)


class PreDecoder:
   def __init__(self, instructionQueue):
      self.B16BlockQueue = deque() # a deque of 16 Byte blocks (i.e., deques of InstrInstances)
      self.instructionQueue = instructionQueue
      self.preDecQueue = deque() # instructions are queued here before they are added to the instruction queue after all stalls have been resolved
      self.stalled = 0
      self.partialInstrI = None

   def cycle(self):
      if not self.stalled:
         if (not self.preDecQueue) and (self.B16BlockQueue or self.partialInstrI): # ToDo: other microarch. than SKL
            if self.partialInstrI is not None:
               self.preDecQueue.append(self.partialInstrI)
               self.partialInstrI = None

            if self.B16BlockQueue:
               curBlock = self.B16BlockQueue[0]

               while curBlock and len(self.preDecQueue) < PreDecode_Width:
                  if instrInstanceCrosses16ByteBoundary(curBlock[0]):
                     break
                  self.preDecQueue.append(curBlock.popleft())

               if len(curBlock) == 1:
                  instrI = curBlock[0]
                  if instrInstanceCrosses16ByteBoundary(instrI):
                     offsetAfterPrefixes = (instrI.address % 16) + instrI.instr.nPrefixes
                     opcodeAfterPrefixes = instrI.instr.opcode[2*instrI.instr.nPrefixes:]
                     onlyPrefixesOr0FInCurBlock = (offsetAfterPrefixes >= 16) or (offsetAfterPrefixes == 15 and opcodeAfterPrefixes.startswith('0F'))
                     if (len(self.preDecQueue) < 5) or onlyPrefixesOr0FInCurBlock:
                        self.partialInstrI = instrI
                        curBlock.popleft()

               if not curBlock:
                  self.B16BlockQueue.popleft()

            self.stalled = sum(3 for ii in self.preDecQueue if ii.instr.lcpStall)

         if not self.stalled and len(self.instructionQueue) + len(self.preDecQueue) < IQ_Width:
            for instrI in self.preDecQueue:
               if instrI.instr.macroFusedWithPrevInstr:
                  continue
               for lamUop in instrI.uops:
                  for fusedUop in lamUop.getFusedUops():
                     for uop in fusedUop.getUnfusedUops():
                        uop.predecoded = clock
               self.instructionQueue.append(instrI)
            self.preDecQueue.clear()

      self.stalled = max(0, self.stalled-1)

   def isEmpty(self):
      return (not self.B16BlockQueue) and (not self.preDecQueue) and (not self.partialInstrI)

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
         if all((u.executed is not None and u.executed < clock) for u in unfusedUops):
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
      self.uops = set()
      self.portUsage = {p:0  for p in allPorts}
      self.uopsDispatchedInPrevCycle = [] # the port usage counter is decreased one cycle after uops are issued
      self.divBusy = 0
      self.readyQueue = {p:[] for p in allPorts}
      self.readyDivUops = []
      self.dependentUops = {}
      self.uopsReadyInCycle = {}

   def isFull(self):
      return len(self.uops) + issue_Width > RS_Width

   def cycle(self, newUops):
      self.divBusy = max(0, self.divBusy-1)
      if clock in self.uopsReadyInCycle:
         for uop in self.uopsReadyInCycle[clock]:
            if uop.divCycles:
               heappush(self.readyDivUops, (uop.idx, uop))
            else:
               heappush(self.readyQueue[uop.actualPort], (uop.idx, uop))
         del self.uopsReadyInCycle[clock]

      self.addNewUops(newUops)
      self.dispatchUops()

   def dispatchUops(self):
      uopsDispatched = []
      for port in allPorts:
         queue = self.readyQueue[port]
         if port == '0' and (not self.divBusy) and self.readyDivUops and ((not self.readyQueue['0']) or self.readyDivUops[0][0] < self.readyQueue['0'][0][0]):
            queue = self.readyDivUops
         if (not queue) and port in ['2', '3']:
            queue = self.readyQueue['2' if port == '3' else '3']
         if not queue:
            continue

         uop = heappop(queue)[1]

         uop.actualPort = port
         uop.dispatched = clock
         uop.executed = self.getFinishTimeEstimate(uop)
         uopsDispatched.append(uop)
         self.divBusy += uop.divCycles

         if uop in self.dependentUops:
            for depUop in self.dependentUops[uop]:
               self.processUop(depUop)
            del self.dependentUops[uop]

         self.uops.remove(uop)

      for uop in self.uopsDispatchedInPrevCycle:
         self.portUsage[uop.actualPort] -= 1
      self.uopsDispatchedInPrevCycle = uopsDispatched


   # adds uops to self.uopsReadyInCycle (and self.readyQueue if applicable?)
   def processUop(self, uop):
      if any((uop2.executed is None) for renInpOp in uop.renamedInputOperands for uop2 in renInpOp.uops):
         return
      try:
         lastUopExecuted = max(uop2.executed for renInpOp in uop.renamedInputOperands for uop2 in renInpOp.uops)
      except ValueError:
         lastUopExecuted = -1
      if lastUopExecuted < uop.issued + issue_dispatch_delay:
         readyCycle = uop.issued + issue_dispatch_delay
      elif (lastUopExecuted == uop.issued + issue_dispatch_delay) or (lastUopExecuted == uop.issued + issue_dispatch_delay + 1):
         readyCycle = lastUopExecuted + 1
      else:
         readyCycle = lastUopExecuted
      readyCycle = max(clock + 1, readyCycle)
      self.uopsReadyInCycle.setdefault(readyCycle, []).append(uop)


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

            uop.actualPort = port
            self.portUsage[port] += 1
            self.uops.add(uop)

            for renInpOp in uop.renamedInputOperands:
               for uop2 in renInpOp.uops:
                  if uop2.executed is None:
                     self.dependentUops.setdefault(uop2, set()).add(uop)

            self.processUop(uop)


   def getFinishTimeEstimate(self, uop):
      finishTime = clock + 1
      for outOp, renOutOp in zip(uop.outputOperands, uop.renamedOutputOperands):
         otherUops = renOutOp.uops # uops that potentially write outOp
         for inpOp, renInpOp in zip(uop.inputOperands, uop.renamedInputOperands):
            minCycle = sys.maxsize
            for uop2 in otherUops:
               if uop2.dispatched is None: continue
               if not renInpOp in uop2.renamedInputOperands: continue
               minCycle = min(minCycle, uop2.dispatched + uop.instr.latencies.get((inpOp, outOp), 1))
            finishTime = max(finishTime, minCycle)
      return finishTime


def getAllPorts():
   if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB']: return [str(i) for i in range(0,6)]
   elif arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL']: return [str(i) for i in range(0,8)]
   elif arch in ['ICL']: return [str(i) for i in range(0,10)]


# must only be called once for a given list of instructions
def adjustLatencies(instructions):
   prevWriteToReg = dict() # reg -> instr
   for instr in instructions:
      for outOp in instr.outputRegOperands:
         if instr.mayBeEliminated: # ToDo: what if not actually eliminated?
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
         if instr.mayBeEliminated:
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
      isBranchInstr = any(True for n, r in instrD.regOperands.items() if ('IP' in r) and ('W' in instrD.rw[n]))

      instruction = None
      for instrData in instrDataDict.get(instrD.iform, []):
         if all(instrD.attributes.get(k, '0') == v for k, v in instrData['attributes'].items()):
            uops = instrData.get('uops', 0)
            retireSlots = instrData.get('retSlots', 0)
            uopsMITE = instrData.get('uopsMITE', 0)
            uopsMS = instrData.get('uopsMS', 0)
            latData = instrData.get('lat', dict())
            portData = instrData.get('ports', {})
            divCycles = instrData.get('divC', {})
            complexDecoder = instrData.get('complDec', False)
            if sameReg:
               uops = instrData.get('uops_SR', uops)
               retireSlots = instrData.get('retSlots_SR', retireSlots)
               uopsMITE = instrData.get('uopsMITE_SR', uopsMITE)
               uopsMS = instrData.get('uopsMS_SR', uopsMS)
               latData = instrData.get('lat_SR', latData)
               portData = instrData.get('ports_SR', portData)
               divCycles = instrData.get('divC_SR',divCycles)
               complexDecoder = instrData.get('complDec_SR', complexDecoder)
            elif usesIndexedAddr:
               uops = instrData.get('uops_I', uops)
               retireSlots = instrData.get('retSlots_I', retireSlots)
               uopsMITE = instrData.get('uopsMITE_I', uopsMITE)
               uopsMS = instrData.get('uopsMS_I', uopsMS)
               portData = instrData.get('ports_I', portData)
               divCycles = instrData.get('divC_I',divCycles)
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

            mayBeEliminated = ('MOV' in instrData['string']) and (not uops) and (len(inputRegOperands) == 1) and (len(outputRegOperands) == 1)
            if mayBeEliminated:
               uops = instrData.get('uops_SR', uops)
               portData = instrData.get('ports_SR', portData)

            instruction = Instr(instrD.asm, instrD.opcode, nPrefixes, instrData['string'], portData, uops, retireSlots, nLaminatedDomainUops, divCycles,
                                inputRegOperands, inputMemOperands, outputRegOperands, outputMemOperands, memAddrOperands, latencies, lcpStall, modifiesStack,
                                complexDecoder, mayBeEliminated, isBranchInstr, instrData.get('macroFusible', set()))
            #print instruction
            break

      if instruction is None:
         instruction = UnknownInstr(instrD.asm, instrD.opcode, nPrefixes)

      # Macro-fusion
      if instructions:
         prevInstr = instructions[-1]
         if instruction.instrStr in prevInstr.macroFusibleWith:
            instruction.macroFusedWithPrevInstr = True
            prevInstr.macroFusedWithNextInstr = True
            instrPorts = instruction.portData.keys()[0]
            if prevInstr.uops == 0:
               prevInstr.uops = instruction.uops
               prevInstr.portData = instruction.portData
            else:
               for p, u in prevInstr.portData.items():
                  if set(instrPorts).issubset(set(p)):
                     del prevInstr.portData[p]
                     prevInstr.portData[instrPorts] = u
                     break

      instructions.append(instruction)
   return instructions


InstrInstance = namedtuple('InstrInstance', ['instr', 'address', 'round', 'uops'])

def split64ByteBlockTo16ByteBlocks(cacheBlock):
   return [[ii for ii in cacheBlock if b*16 <= ii.address % 64 < (b+1)*16 ] for b in range(0,4)]

def split32ByteBlockTo16ByteBlocks(B32Block):
   return [[ii for ii in B32Block if b*16 <= ii.address % 32 < (b+1)*16 ] for b in range(0,2)]

def split64ByteBlockTo32ByteBlocks(cacheBlock):
   return [[ii for ii in cacheBlock if b*32 <= ii.address % 64 < (b+1)*32 ] for b in range(0,2)]

def instrInstanceCrosses16ByteBoundary(instrI):
   instrLen = len(instrI.instr.opcode)/2
   return (instrI.address % 16) + instrLen > 16

# returns list of instrInstances corresponding to a 64-Byte cache block
def CacheBlockGenerator(instructions, unroll):
   cacheBlock = []
   nextAddr = 0
   for instr, uops, rnd in UopGenerator(instructions):
      cacheBlock.append(InstrInstance(instr, nextAddr, rnd, uops))

      if (not unroll) and instr == instructions[-1]:
         yield cacheBlock
         cacheBlock = []
         nextAddr = 0
         continue

      prevAddr = nextAddr
      nextAddr = prevAddr + len(instr.opcode)/2
      if prevAddr / 64 != nextAddr / 64:
         yield cacheBlock
         cacheBlock = []


# returns cache blocks for one round (without unrolling)
def CacheBlocksForNextRoundGenerator(instructions):
   cacheBlocks = []
   prevRnd = 0
   for cacheBlock in CacheBlockGenerator(instructions, unroll=False):
      curRnd = cacheBlock[-1].round
      if prevRnd != curRnd:
         yield cacheBlocks
         cacheBlocks = []
         prevRnd = curRnd
      cacheBlocks.append(cacheBlock)


# repeatedly iterates over instructions; in each step, returns uops for one instruction
def UopGenerator(instructions):
   stackPtrImplicitlyModified = False
   for rnd in count():
      for instr in instructions:
         if instr.macroFusedWithPrevInstr:
            yield (instr, [], rnd)
            continue

         allRegOperands = instr.inputRegOperands + instr.outputRegOperands + instr.memAddrOperands

         stackSynchUop = None
         if stackPtrImplicitlyModified and any((getCanonicalReg(op.reg) == 'RSP') for op in allRegOperands):
            stackSynchUop = StackSynchUop(instr, rnd)
            stackPtrImplicitlyModified = False

         if instr.modifiesStack:
            stackPtrImplicitlyModified = True

         allInputOperands = instr.inputRegOperands + instr.memAddrOperands + instr.inputMemOperands

         AGUPcs = []
         storeDataPcs = []
         nonMemPcs = []
         for pc, n in instr.portData.items():
            ports = list(pc)
            if any((p in ports) for p in ['2', '3', '7', '8']):
               AGUPcs.extend([ports]*n)
            elif any((p in ports) for p in ['4', '9']):
               storeDataPcs.extend([ports]*n)
            else:
               nonMemPcs.extend([ports]*n)

         unfusedUops = deque()

         for pc in AGUPcs:
            applicableInputOperands = instr.memAddrOperands + instr.inputMemOperands # ToDo: distinguish between load and store operations
            applicableOutputOperands = instr.outputRegOperands + instr.outputMemOperands
            unfusedUops.append(Uop(instr, rnd, pc, applicableInputOperands, applicableOutputOperands))
         for pc in storeDataPcs:
            applicableInputOperands = allInputOperands
            applicableOutputOperands = instr.outputMemOperands
            unfusedUops.append(Uop(instr, rnd, pc, applicableInputOperands, applicableOutputOperands))

         lat1OutputRegs = [] # output register operands that have a latency of at most 1 from all input registers
         lat1InputOperands = set() # input operands that have a latency of 1 to the output operands in lat1OutputRegs
         for outOp in instr.outputRegOperands:
            if all(instr.latencies.get((inOp, outOp), 2) <= 1 for inOp in allInputOperands):
               lat1OutputRegs.append(outOp)
               lat1InputOperands.update(inOp for inOp in allInputOperands if instr.latencies.get((inOp, outOp), 2) == 1)

         nonLat1OutputOperands = instr.outputRegOperands + instr.outputMemOperands
         divCyclesAdded = False
         for i, pc in enumerate(nonMemPcs):
            if (i == 0) and (len(nonMemPcs) > 1) and lat1OutputRegs:
               applicableInputOperands = list(lat1InputOperands)
               applicableOutputOperands = lat1OutputRegs
               nonLat1OutputOperands = [op for op in nonLat1OutputOperands if not op in lat1OutputRegs]
            else:
               applicableInputOperands = allInputOperands
               applicableOutputOperands = nonLat1OutputOperands

            divCycles = 0
            if instr.divCycles and not divCyclesAdded and pc == ['0']:
               divCycles = instr.divCycles
               divCyclesAdded = True

            unfusedUops.append(Uop(instr, rnd, pc, applicableInputOperands, applicableOutputOperands, divCycles))

         for _ in range(0, instr.retireSlots - len(unfusedUops)):
            uop = Uop(instr, rnd, None, [], [])
            unfusedUops.append(uop)

         fusedDomainUops = deque()
         for _ in range(0, instr.retireSlots-1):
            fusedDomainUops.append(FusedUop([unfusedUops.popleft()]))
         fusedDomainUops.append(FusedUop(unfusedUops))
         if stackSynchUop is not None:
            fusedDomainUops.append(stackSynchUop)

         laminatedDomainUops = []
         for _ in range(0, instr.nLaminatedDomainUops-1):
            laminatedDomainUops.append(LaminatedUop([fusedDomainUops.popleft()]))
         laminatedDomainUops.append(LaminatedUop(fusedDomainUops))

         yield (instr, laminatedDomainUops, rnd)


def printPortUsage(instructions, uopsForRound):
   formatStr = '|' + '{:^9}|'*(len(allPorts)+1)

   print '-'*(1+10*(len(allPorts)+1))
   print formatStr.format('Uops', *allPorts)
   print '-'*(1+10*(len(allPorts)+1))
   portUsageC = Counter(uop.actualPort for uopsDict in uopsForRound for uops in uopsDict.values() for uop in uops)
   portUsageL = [('{:.2f}'.format(float(portUsageC[p])/len(uopsForRound)) if p in portUsageC else '') for p in allPorts]
   #print formatStr.format(str(sum(len(uops) for uops in uopsForRound[0].values())), *portUsageL)
   print formatStr.format(str(sum(instr.uops for instr in instructions if not instr.macroFusedWithPrevInstr)), *portUsageL)
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
      elif instr.macroFusedWithPrevInstr:
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

   nRows = sum(max(len(uops),1) for uops in uopsForRound[0].values())
   for rnd, uopsDict in enumerate(uopsForRound):
      table.append('<tr style="border-top: 2px solid black">')
      table.append('<td rowspan="{}">{}</td>'.format(nRows, rnd))
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

         if not uopsDict[instr]:
            table.append('<td>-</td><td>-</td>')
            table.append('<td></td>'*(maxCycle+1))
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
   parser.add_argument("-loop", help="loop", action='store_true')
   args = parser.parse_args()

   global arch, allPorts
   arch = args.arch
   allPorts = getAllPorts()

   instrDataDict = importlib.import_module('instrData.'+arch).instrData

   instructions = getInstructions(args.filename, args.raw, args.iacaMarkers, instrDataDict)
   lastApplicableInstr = [instr for instr in instructions if not instr.macroFusedWithPrevInstr][-1] # ignore macro-fused instr.
   adjustLatencies(instructions)
   #print instructions

   global clock
   clock = 0

   #uopGenerator = UopGenerator(instructions)
   retireQueue = deque()
   rb = ReorderBuffer(retireQueue)
   scheduler = Scheduler()

   frontEnd = FrontEnd(instructions, rb, scheduler, not args.loop)
   #   uopSource = Decoder(uopGenerator, IDQ)
   #else:
   #   uopSource = DSB(uopGenerator, IDQ)


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

