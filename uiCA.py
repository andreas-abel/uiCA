#!/usr/bin/env python3

import argparse
import importlib
import os
import re
from collections import Counter, deque, namedtuple, OrderedDict
from concurrent import futures
from heapq import heappop, heappush
from itertools import count, repeat
from typing import List, Dict, NamedTuple, Optional

import random
random.seed(0)

import xed

from facile import *
from instrData.uArchInfo import allPorts, ALUPorts
from instructions import *
from microArchConfigs import MicroArchConfig, MicroArchConfigs
from utils import *
from x64_lib import *

class UopProperties:
   def __init__(self, instr, possiblePorts, inputOperands, outputOperands, latencies, divCycles=0, isLoadUop=False, isStoreAddressUop=False, memAddr=None,
                isStoreDataUop=False, isFirstUopOfInstr=False, isLastUopOfInstr=False, isRegMergeUop=False):
      self.instr = instr
      self.possiblePorts = possiblePorts
      self.inputOperands = inputOperands
      self.outputOperands = outputOperands
      self.latencies = latencies # latencies[outOp] = x
      self.divCycles = divCycles
      self.isLoadUop = isLoadUop
      self.isStoreAddressUop = isStoreAddressUop
      self.memAddr = memAddr
      self.isStoreDataUop = isStoreDataUop
      self.isFirstUopOfInstr = isFirstUopOfInstr
      self.isLastUopOfInstr = isLastUopOfInstr
      self.isRegMergeUop = isRegMergeUop

   def __str__(self):
      return 'UopProperties(ports: {}, in: {}, out: {}, lat: {})'.format(self.possiblePorts, self.inputOperands, self.outputOperands, self.latencies)

class Uop:
   idx_iter = count()

   def __init__(self, prop, instrI):
      self.idx = next(self.idx_iter)
      self.prop: UopProperties = prop
      self.instrI: InstrInstance = instrI
      self.fusedUop: Optional[FusedUop] = None # fused-domain uop that contains this uop
      self.actualPort = None
      self.eliminated = False
      self.renamedInputOperands: List[RenamedOperand] = []
      self.renamedOutputOperands: List[RenamedOperand] = []
      self.storeBufferEntry = None
      self.latReducedDueToFastPtrChasing = False
      self.readyForDispatch = None
      self.dispatched = None
      self.executed = None

   def getUnfusedUops(self):
      return [self]

   def __str__(self):
      return 'Uop(idx: {}, rnd: {}, p: {})'.format(self.idx, self.instrI.rnd, self.actualPort)


class FusedUop:
   def __init__(self, uops: List[Uop]):
      self.__uops = uops
      for uop in uops:
         uop.fusedUop = self
      self.laminatedUop: Optional[LaminatedUop] = None # laminated-domain uop that contains this
      self.issued = None # cycle in which this uop was issued
      self.retired = None # cycle in which this uop was retired
      self.retireIdx = None # how many other uops were already retired in the same cycle

   def getUnfusedUops(self):
      return self.__uops


class LaminatedUop:
   def __init__(self, fusedUops: List[FusedUop]):
      self.__fusedUops = fusedUops
      for fUop in fusedUops:
         fUop.laminatedUop = self
      self.addedToIDQ = None # cycle in which this uop was added to the IDQ
      self.uopSource = None # MITE, DSB, MS, LSD, or SE

   def getFusedUops(self):
      return self.__fusedUops

   def getUnfusedUops(self):
      return [uop for fusedUop in self.getFusedUops() for uop in fusedUop.getUnfusedUops()]


class StackSyncUop(Uop):
   def __init__(self, instrI, uArchConfig):
      inOp = RegOperand('RSP')
      outOp = RegOperand('RSP')
      prop = UopProperties(instrI.instr, ALUPorts[uArchConfig.name], [inOp], [outOp], {outOp: 1}, isFirstUopOfInstr=True)
      Uop.__init__(self, prop, instrI)


class StoreBufferEntry:
   def __init__(self, abstractAddress):
      self.abstractAddress = abstractAddress # (base, index, scale, disp)
      self.uops = [] # uops that write to this entry
      self.addressReadyCycle = None
      self.dataReadyCycle = None

class RenamedOperand:
   def __init__(self, nonRenamedOperand=None, uop=None, ready=None):
      self.nonRenamedOperand = nonRenamedOperand
      self.uop = uop # uop that writes this operand
      self.__ready = ready # cycle in which operand becomes ready

   def getReadyCycle(self):
      if self.__ready is not None:
         return self.__ready

      if self.uop.dispatched is None:
         return None

      lat = self.uop.prop.latencies.get(self.nonRenamedOperand, 1)
      if self.uop.latReducedDueToFastPtrChasing:
         lat -= 1

      if self.uop.prop.isLoadUop and (self.uop.storeBufferEntry is not None):
         sb = self.uop.storeBufferEntry
         if (sb.addressReadyCycle is None) or (sb.dataReadyCycle is None):
            return None
         memReady = max(sb.addressReadyCycle, sb.dataReadyCycle) + 4 # ToDo
         self.__ready = max(self.uop.dispatched + lat, memReady)
      else:
         self.__ready = self.uop.dispatched + lat

      return self.__ready


RenameDictEntry = namedtuple('RenameDictEntry', ['renamedOp', 'renamedByElim32BitMove'])
class Renamer:
   def __init__(self, IDQ, reorderBuffer, uArchConfig: MicroArchConfig, initPolicy):
      self.IDQ = IDQ
      self.reorderBuffer = reorderBuffer
      self.uArchConfig = uArchConfig
      self.absValGen = AbstractValueGenerator(initPolicy)

      self.renameDict = {}

      # renamed operands written by current instr.
      self.curInstrRndRenameDict = {}
      self.curInstrPseudoOpDict = {}

      self.nGPRMoveElimInCycle = {}
      self.multiUseGPRDict = {}
      self.multiUseGPRDictUseInCycle = {}

      self.nSIMDMoveElimInCycle = {}
      self.multiUseSIMDDict = {}
      self.multiUseSIMDDictUseInCycle = {}

      self.renamerActiveCycle = 0

      self.curStoreBufferEntry = None
      self.storeBufferEntryDict = {}

      self.lastRegMergeIssued = None # last uop for which register merge uops were issued

   def cycle(self):
      self.renamerActiveCycle += 1

      renamerUops = []
      while self.IDQ:
         lamUop = self.IDQ[0]

         firstUnfusedUop = lamUop.getUnfusedUops()[0]
         regMergeProps = firstUnfusedUop.prop.instr.regMergeUopPropertiesList
         if firstUnfusedUop.prop.isFirstUopOfInstr and regMergeProps:
            if renamerUops:
               break
            if self.lastRegMergeIssued != firstUnfusedUop:
               for mergeProp in regMergeProps:
                  mergeUop = FusedUop([Uop(mergeProp, firstUnfusedUop.instrI)])
                  renamerUops.append(mergeUop)
                  firstUnfusedUop.instrI.regMergeUops.append(LaminatedUop([mergeUop]))
               self.lastRegMergeIssued = firstUnfusedUop
               break

         if firstUnfusedUop.prop.isFirstUopOfInstr and firstUnfusedUop.prop.instr.isSerializingInstr and not self.reorderBuffer.isEmpty():
            break
         fusedUops = lamUop.getFusedUops()
         if len(renamerUops) + len(fusedUops) > self.uArchConfig.issueWidth:
            break
         renamerUops.extend(fusedUops)
         self.IDQ.popleft()

      nGPRMoveElim = 0
      nSIMDMoveElim = 0

      for fusedUop in renamerUops:
         for uop in fusedUop.getUnfusedUops():
            if uop.prop.instr.mayBeEliminated and (not uop.prop.isRegMergeUop) and (not isinstance(uop, StackSyncUop)):
               canonicalInpReg = getCanonicalReg(uop.prop.instr.inputRegOperands[0].reg)

               if (canonicalInpReg in GPRegs):
                  if self.uArchConfig.moveEliminationGPRSlots == 'unlimited':
                     nGPRMoveElimPossible = 1
                  else:
                     nGPRMoveElimPossible = (self.uArchConfig.moveEliminationGPRSlots - nGPRMoveElim
                           - sum(self.nGPRMoveElimInCycle.get(self.renamerActiveCycle - i, 0) for i in range(1, self.uArchConfig.moveEliminationPipelineLength))
                           - self.multiUseGPRDictUseInCycle.get(self.renamerActiveCycle - self.uArchConfig.moveEliminationPipelineLength, 0))
                  if nGPRMoveElimPossible > 0:
                     uop.eliminated = True
                     nGPRMoveElim += 1
               elif ('MM' in canonicalInpReg):
                  if self.uArchConfig.moveEliminationSIMDSlots == 'unlimited':
                     nSIMDMoveElimPossible = 1
                  else:
                     nSIMDMoveElimPossible = (self.uArchConfig.moveEliminationSIMDSlots - nSIMDMoveElim
                           - sum(self.nSIMDMoveElimInCycle.get(self.renamerActiveCycle - i, 0) for i in range(1, self.uArchConfig.moveEliminationPipelineLength))
                           - self.multiUseSIMDDictUseInCycle.get(self.renamerActiveCycle - self.uArchConfig.moveEliminationPipelineLength, 0))
                  if nSIMDMoveElimPossible > 0:
                     uop.eliminated = True
                     nSIMDMoveElim += 1

      if (nGPRMoveElim == 0) and (not self.uArchConfig.moveEliminationGPRAllAliasesMustBeOverwritten):
         for k, v in list(self.multiUseGPRDict.items()):
            if len(v) <= 1:
               del self.multiUseGPRDict[k]
      if nSIMDMoveElim == 0:
         for k, v in list(self.multiUseSIMDDict.items()):
            if len(v) <= 1:
               del self.multiUseSIMDDict[k]

      for fusedUop in renamerUops:
         for uop in fusedUop.getUnfusedUops():
            if uop.eliminated:
               is32BitMove = (getRegSize(uop.prop.instr.outputRegOperands[0].reg) == 32)
               canonicalInpReg = getCanonicalReg(uop.prop.instr.inputRegOperands[0].reg)
               canonicalOutReg = getCanonicalReg(uop.prop.instr.outputRegOperands[0].reg)

               if (canonicalInpReg in GPRegs):
                  curMultiUseDict = self.multiUseGPRDict
               else:
                  curMultiUseDict = self.multiUseSIMDDict

               entry = self.renameDict.setdefault(canonicalInpReg, RenameDictEntry(RenamedOperand(ready=-1), False))
               self.curInstrRndRenameDict[canonicalOutReg] = RenameDictEntry(entry.renamedOp, is32BitMove)
               if not entry.renamedOp in curMultiUseDict:
                  curMultiUseDict[entry.renamedOp] = set()
               curMultiUseDict[entry.renamedOp].update([canonicalInpReg, canonicalOutReg])

               key = self.getRenameDictKey(uop.prop.instr.outputRegOperands[0])
               self.absValGen.setAbstractValueForCurInstr(key, uop.prop.instr)
            else:
               if uop.prop.instr.uops or isinstance(uop, StackSyncUop):
                  if uop.prop.isStoreAddressUop:
                     key = self.getStoreBufferKey(uop.prop.memAddr)
                     self.curStoreBufferEntry = StoreBufferEntry(key)
                     self.storeBufferEntryDict[key] = self.curStoreBufferEntry
                  if uop.prop.isStoreAddressUop or uop.prop.isStoreDataUop:
                     uop.storeBufferEntry = self.curStoreBufferEntry
                     self.curStoreBufferEntry.uops.append(uop)
                  if uop.prop.isLoadUop:
                     key = self.getStoreBufferKey(uop.prop.memAddr)
                     baseReg = uop.prop.memAddr.get('base')
                     baseEntry = self.renameDict.get(getCanonicalReg(baseReg)) if baseReg else None
                     baseInstr = baseEntry.renamedOp.uop.prop.instr if (baseEntry and baseEntry.renamedOp.uop) else None
                     indexReg = uop.prop.memAddr.get('index')
                     indexEntry = self.renameDict.get(getCanonicalReg(indexReg)) if indexReg else None
                     indexInstr = indexEntry.renamedOp.uop.prop.instr if (indexEntry and indexEntry.renamedOp.uop) else None
                     uop.latReducedDueToFastPtrChasing = latReducedDueToFastPtrChasing(self.uArchConfig, uop.prop.memAddr, baseInstr, indexInstr,
                                                                                       (baseEntry and baseEntry.renamedByElim32BitMove))
                     uop.storeBufferEntry = self.storeBufferEntryDict.get(key, None)

                  for inpOp in uop.prop.inputOperands:
                     if isinstance(inpOp, PseudoOperand):
                        renOp = self.curInstrPseudoOpDict[inpOp]
                     else:
                        key = self.getRenameDictKey(inpOp)
                        renOp = self.renameDict.setdefault(key, RenameDictEntry(RenamedOperand(ready=-1), False)).renamedOp
                     uop.renamedInputOperands.append(renOp)
                  for outOp in uop.prop.outputOperands:
                     renOp = RenamedOperand(outOp, uop)
                     uop.renamedOutputOperands.append(renOp)
                     if isinstance(outOp, PseudoOperand):
                        self.curInstrPseudoOpDict[outOp] = renOp
                     else:
                        key = self.getRenameDictKey(outOp)
                        self.curInstrRndRenameDict[key] = RenameDictEntry(renOp, False)
                        if isinstance(outOp, RegOperand):
                           self.absValGen.setAbstractValueForCurInstr(key, uop.prop.instr)
               else:
                  # e.g., xor rax, rax
                  for op in uop.prop.instr.outputRegOperands:
                     self.curInstrRndRenameDict[getCanonicalReg(op.reg)] = RenameDictEntry(RenamedOperand(uop=uop, ready=-1), False)

            if uop.prop.isLastUopOfInstr or uop.prop.isRegMergeUop or isinstance(uop, StackSyncUop):
               for key in self.curInstrRndRenameDict:
                  if key in self.renameDict:
                     prevRenOp = self.renameDict[key].renamedOp
                     if (not uop.eliminated) or (prevRenOp != self.curInstrRndRenameDict[key].renamedOp):
                        if (key in GPRegs) and (prevRenOp in self.multiUseGPRDict):
                           self.multiUseGPRDict[prevRenOp].remove(key)
                        elif (type(key) == str) and ('MM' in key) and (prevRenOp in self.multiUseSIMDDict):
                           if self.multiUseSIMDDict[prevRenOp]:
                              self.multiUseSIMDDict[prevRenOp].remove(key)

               self.renameDict.update(self.curInstrRndRenameDict)
               self.absValGen.finishCurInstr()
               self.curInstrRndRenameDict.clear()
               self.curInstrPseudoOpDict.clear()

      self.nGPRMoveElimInCycle[self.renamerActiveCycle] = nGPRMoveElim
      self.nSIMDMoveElimInCycle[self.renamerActiveCycle] = nSIMDMoveElim

      for k, v in list(self.multiUseGPRDict.items()):
         if len(v) == 0:
            del self.multiUseGPRDict[k]
      if self.multiUseGPRDict:
         self.multiUseGPRDictUseInCycle[self.renamerActiveCycle] = len(self.multiUseGPRDict)

      for k, v in list(self.multiUseSIMDDict.items()):
         if len(v) == 0:
            del self.multiUseSIMDDict[k]
      if self.multiUseSIMDDict:
         self.multiUseSIMDDictUseInCycle[self.renamerActiveCycle] = len(self.multiUseSIMDDict)

      return renamerUops

   def getRenameDictKey(self, op):
      if isinstance(op, RegOperand):
         return getCanonicalReg(op.reg)
      elif isinstance(op, FlagOperand):
         return op.flags
      else:
         return None

   def getStoreBufferKey(self, memAddr):
      if memAddr is None:
         return None
      return (self.absValGen.getAbstractValueForReg(memAddr.get('base')), self.absValGen.getAbstractValueForReg(memAddr.get('index')),
              memAddr.get('scale'), memAddr.get('disp'))


class FrontEnd:
   def __init__(self, instructions: List[Instr], reorderBuffer, scheduler, uArchConfig: MicroArchConfig,
                unroll, alignmentOffset, initPolicy, perfEvents, simpleFrontEnd=False):
      self.IDQ = deque()
      self.renamer = Renamer(self.IDQ, reorderBuffer, uArchConfig, initPolicy)
      self.reorderBuffer = reorderBuffer
      self.scheduler = scheduler
      self.uArchConfig = uArchConfig
      self.unroll = unroll
      self.alignmentOffset = alignmentOffset
      self.perfEvents = perfEvents

      self.MS = MicrocodeSequencer(self.uArchConfig)

      self.instructionQueue = deque()
      self.preDecoder = PreDecoder(self.instructionQueue, self.uArchConfig)
      self.decoder = Decoder(self.instructionQueue, self.MS, self.uArchConfig)

      self.RSPOffset = 0

      self.allGeneratedInstrInstances: List[InstrInstance] = []

      self.DSB = DSB(self.MS, self.uArchConfig)
      self.addressesInDSB = set()

      self.LSDUnrollCount = 1

      if simpleFrontEnd:
         self.uopSource = None
      else:
         self.uopSource = 'MITE'

      if unroll or simpleFrontEnd:
         self.cacheBlockGenerator = CacheBlockGenerator(instructions, True, self.alignmentOffset)
      else:
         self.cacheBlocksForNextRoundGenerator = CacheBlocksForNextRoundGenerator(instructions, self.alignmentOffset)
         cacheBlocksForFirstRound = next(self.cacheBlocksForNextRoundGenerator)

         if self.uArchConfig.DSBBlockSize == 32:
            allBlocksCanBeCached = all(canBeInDSB(block, uArchConfig.DSBBlockSize) for cb in cacheBlocksForFirstRound
                                       for block in split64ByteBlockTo32ByteBlocks(cb) if block)
         else:
            allBlocksCanBeCached = all(canBeInDSB(block, uArchConfig.DSBBlockSize) for block in cacheBlocksForFirstRound)

         allInstrsCanBeUsedByLSD = all(instrI.instr.canBeUsedByLSD() for cb in cacheBlocksForFirstRound for instrI in cb)
         nUops = sum(len(instrI.uops) for cb in cacheBlocksForFirstRound for instrI in cb)
         if allBlocksCanBeCached and self.uArchConfig.LSDEnabled and allInstrsCanBeUsedByLSD and (nUops <= self.uArchConfig.IDQWidth):
            self.uopSource = 'LSD'
            self.LSDUnrollCount = self.uArchConfig.LSDUnrolling.get(nUops, 1)
            for cacheBlock in cacheBlocksForFirstRound + [cb for _ in range(0, self.LSDUnrollCount-1) for cb in next(self.cacheBlocksForNextRoundGenerator)]:
               self.addNewCacheBlock(cacheBlock)
         else:
            self.findCacheableAddresses(cacheBlocksForFirstRound)
            for cacheBlock in cacheBlocksForFirstRound:
               self.addNewCacheBlock(cacheBlock)
            if self.alignmentOffset in self.addressesInDSB:
               self.uopSource = 'DSB'

   def cycle(self, clock):
      issueUops = []
      if not self.reorderBuffer.isFull() and not self.scheduler.isFull(): # len(self.IDQ) >= uArchConfig.issueWidth and the first check seems to be wrong, but leads to better results
         issueUops = self.renamer.cycle()

      for fusedUop in issueUops:
         fusedUop.issued = clock

      self.reorderBuffer.cycle(clock, issueUops)
      self.scheduler.cycle(clock, issueUops)

      if self.reorderBuffer.isFull():
         self.perfEvents.setdefault(clock, {})['RBFull'] = 1
      if self.scheduler.isFull():
         self.perfEvents.setdefault(clock, {})['RSFull'] = 1
      if len(self.instructionQueue) + self.uArchConfig.preDecodeWidth > self.uArchConfig.IQWidth:
         self.perfEvents.setdefault(clock, {})['IQFull'] = 1

      if len(self.IDQ) + self.uArchConfig.DSBWidth > self.uArchConfig.IDQWidth:
         self.perfEvents.setdefault(clock, {})['IDQFull'] = 1
         return

      if self.uopSource is None:
         while len(self.IDQ) < self.uArchConfig.issueWidth:
            for instrI in next(self.cacheBlockGenerator):
               self.allGeneratedInstrInstances.append(instrI)
               for lamUop in instrI.uops:
                  self.addStackSyncUop(clock, lamUop.getUnfusedUops()[0])
                  for uop in lamUop.getUnfusedUops():
                     self.IDQ.append(LaminatedUop([FusedUop([uop])]))
      elif self.uopSource == 'LSD':
         if not self.IDQ:
            for _ in range(0, self.LSDUnrollCount):
               for cacheBlock in next(self.cacheBlocksForNextRoundGenerator):
                  self.addNewCacheBlock(cacheBlock)
      else:
         # add new cache blocks
         while len(self.DSB.DSBBlockQueue) < 2 and len(self.preDecoder.B16BlockQueue) < 4:
            if self.unroll:
               self.addNewCacheBlock(next(self.cacheBlockGenerator))
            else:
               for cacheBlock in next(self.cacheBlocksForNextRoundGenerator):
                  self.addNewCacheBlock(cacheBlock)

         # add new uops to IDQ
         newUops = []
         if self.MS.isBusy():
            newUops = self.MS.cycle()
         elif self.uopSource == 'MITE':
            self.preDecoder.cycle(clock)
            newInstrIUops = self.decoder.cycle(clock)
            newUops = [u for _, u in newInstrIUops if u is not None]
            if not self.unroll and newInstrIUops:
               curInstrI = newInstrIUops[-1][0]
               if curInstrI.instr.isLastDecodedInstr() and (curInstrI.instr.isBranchInstr or curInstrI.instr.macroFusedWithNextInstr):
                  if self.alignmentOffset in self.addressesInDSB:
                     self.uopSource = 'DSB'
         elif self.uopSource == 'DSB':
            newInstrIUops = self.DSB.cycle()
            newUops = [u for _, u in newInstrIUops if u is not None]
            if newUops and newUops[-1].getUnfusedUops()[-1].prop.isLastUopOfInstr:
               curInstrI = newInstrIUops[-1][0]
               if curInstrI.instr.isLastDecodedInstr():
                  nextAddr = self.alignmentOffset
               else:
                  nextAddr = curInstrI.address + (len(curInstrI.instr.opcode) // 2)
               if nextAddr not in self.addressesInDSB:
                  self.uopSource = 'MITE'

         for lamUop in newUops:
            self.addStackSyncUop(clock, lamUop.getUnfusedUops()[0])
            self.IDQ.append(lamUop)
            lamUop.addedToIDQ = clock

   def findCacheableAddresses(self, cacheBlocksForFirstRound):
      for cacheBlock in cacheBlocksForFirstRound:
         if self.uArchConfig.DSBBlockSize == 32:
            splitCacheBlocks = [block for block in split64ByteBlockTo32ByteBlocks(cacheBlock) if block]
            if self.uArchConfig.both32ByteBlocksMustBeCacheable and any((not canBeInDSB(block, self.uArchConfig.DSBBlockSize)) for block in splitCacheBlocks):
               return
         else:
            splitCacheBlocks = [cacheBlock]

         for block in splitCacheBlocks:
            if canBeInDSB(block, self.uArchConfig.DSBBlockSize):
               for instrI in block:
                  self.addressesInDSB.add(instrI.address)
            else:
               return

   def addNewCacheBlock(self, cacheBlock):
      self.allGeneratedInstrInstances.extend(cacheBlock)
      if self.uopSource == 'LSD':
         for instrI in cacheBlock:
            self.IDQ.extend(instrI.uops)
            instrI.source = 'LSD'
            for uop in instrI.uops:
               uop.uopSource = 'LSD'
      else:
         if self.uArchConfig.DSBBlockSize == 32:
            blocks = split64ByteBlockTo32ByteBlocks(cacheBlock)
         else:
            blocks = [cacheBlock]
         for block in blocks:
            if not block: continue
            if block[0].address in self.addressesInDSB:
               for instrI in block:
                  instrI.source = 'DSB'
               self.DSB.DSBBlockQueue += getDSBBlocks(block)
            else:
               for instrI in block:
                  instrI.source = 'MITE'
               if self.uArchConfig.DSBBlockSize == 32:
                  B16Blocks = split32ByteBlockTo16ByteBlocks(block)
               else:
                  B16Blocks = split64ByteBlockTo16ByteBlocks(block)
               for B16Block in B16Blocks:
                  if not B16Block: continue
                  self.preDecoder.B16BlockQueue.append(deque(B16Block))
                  lastInstrI = B16Block[-1]
                  if lastInstrI.instr.isBranchInstr and (lastInstrI.address % 16) + (len(lastInstrI.instr.opcode) // 2) > 16:
                     # branch instr. ends in next block
                     self.preDecoder.B16BlockQueue.append(deque())

   def addStackSyncUop(self, clock, uop):
      if not uop.prop.isFirstUopOfInstr:
         return

      instr = uop.prop.instr
      requiresSyncUop = False

      if self.RSPOffset and any((getCanonicalReg(op.reg) == 'RSP') for op in instr.inputRegOperands+instr.memAddrOperands if not op.isImplicitStackOperand):
         requiresSyncUop = True
         self.RSPOffset = 0

      self.RSPOffset += instr.implicitRSPChange
      if self.RSPOffset > 192:
         requiresSyncUop = True
         self.RSPOffset = 0

      if any((getCanonicalReg(op.reg) == 'RSP') for op in instr.outputRegOperands):
         self.RSPOffset = 0

      if requiresSyncUop:
         stackSyncUop = StackSyncUop(uop.instrI, self.uArchConfig)
         lamUop = LaminatedUop([FusedUop([stackSyncUop])])
         self.IDQ.append(lamUop)
         lamUop.addedToIDQ = clock
         lamUop.uopSource = 'SE'
         uop.instrI.stackSyncUops.append(lamUop)


DSBEntry = namedtuple('DSBEntry', ['slot', 'instrI', 'uop', 'MSUops', 'requiresExtraEntry'])

class DSB:
   def __init__(self, MS, uArchConfig: MicroArchConfig):
      self.MS = MS
      self.DSBBlockQueue = deque()
      self.uArchConfig = uArchConfig
      self.delayInPrevCycle = False

   def cycle(self):
      retList = []
      DSBBlock = self.DSBBlockQueue[0]
      newDSBBlockStarted = (DSBBlock[0].slot == 0)
      secondDSBBlockLoaded = False
      remainingSlots = self.uArchConfig.DSBWidth
      delayInPrevCycle = self.delayInPrevCycle
      self.delayInPrevCycle = False
      while remainingSlots > 0:
         if not DSBBlock:
            if (not secondDSBBlockLoaded) and self.DSBBlockQueue and (not self.DSBBlockQueue[0][-1].MSUops):
               secondDSBBlockLoaded = True
               DSBBlock = self.DSBBlockQueue[0]
               prevInstrI = retList[-1][0]
               if ((prevInstrI.address + len(prevInstrI.instr.opcode)/2 != DSBBlock[0].instrI.address) and
                     not prevInstrI.instr.isLastDecodedInstr()):
                  # next instr not in DSB
                  return retList
            else:
               return retList

         entry = DSBBlock[0]

         if entry.requiresExtraEntry and (remainingSlots < 2):
            return retList

         if entry.uop:
            retList.append((entry.instrI, entry.uop))
            entry.uop.uopSource = 'DSB'
            if entry.requiresExtraEntry:
               remainingSlots = 0
               self.delayInPrevCycle = True
            else:
               remainingSlots -= 1
         if entry.MSUops:
            self.MS.addUops(entry.MSUops, 'DSB')
            remainingSlots = 0

         DSBBlock.popleft()
         if not DSBBlock:
            self.DSBBlockQueue.popleft()
            if remainingSlots and self.DSBBlockQueue and (self.uArchConfig.DSBWidth == 6):
               nextInstrAddr = self.DSBBlockQueue[0][0].instrI.address
               nextInstrInSameMemoryBlock = (nextInstrAddr//self.uArchConfig.DSBBlockSize == entry.instrI.address//self.uArchConfig.DSBBlockSize)
               if (self.uArchConfig.DSBBlockSize == 32) and nextInstrInSameMemoryBlock and entry.instrI.instr.isLastDecodedInstr() and (not delayInPrevCycle):
                  remainingSlots = 0
               elif not nextInstrInSameMemoryBlock:
                  if newDSBBlockStarted:
                     if len(retList) in [1, 2]:
                        remainingSlots = 4
                     elif len(retList) in [3, 4]:
                        remainingSlots = 2
                     elif len(retList) == 5:
                        remainingSlots = 1
                  elif entry.instrI.instr.isLastDecodedInstr():
                     if (len(retList) == 1) or ((len(retList) == 2) and (entry.slot >= 4)):
                        remainingSlots = 4
                     else:
                        remainingSlots = min(remainingSlots, 2)

      return retList


class MicrocodeSequencer:
   def __init__(self, uArchConfig: MicroArchConfig):
      self.uArchConfig = uArchConfig
      self.uopQueue = deque()
      self.stalled = 0
      self.postStall = 0

   def cycle(self):
      uops = []
      if self.stalled:
         self.stalled -= 1
      elif self.uopQueue:
         while self.uopQueue and len(uops) < 4:
            uops.append(self.uopQueue.popleft())
         if not self.uopQueue:
            self.stalled = self.postStall
      return uops

   def addUops(self, uops, prevUopSource):
      self.uopQueue.extend(uops)
      for lamUop in uops:
         lamUop.uopSource = 'MS'
      if prevUopSource == 'MITE':
         self.stalled = 1
         self.postStall = 1
      elif prevUopSource == 'DSB':
         self.stalled = self.uArchConfig.DSB_MS_Stall
         self.postStall = 0

   def isBusy(self):
      return (len(self.uopQueue) > 0) or self.stalled


class Decoder:
   def __init__(self, instructionQueue, MS: MicrocodeSequencer, uArchConfig: MicroArchConfig):
      self.instructionQueue = instructionQueue
      self.MS = MS
      self.uArchConfig = uArchConfig

   def cycle(self, clock):
      uopsList = []
      nDecodedInstrs = 0
      remainingDecoderSlots = self.uArchConfig.nDecoders
      while self.instructionQueue:
         instrI: InstrInstance = self.instructionQueue[0]
         if instrI.instr.macroFusedWithPrevInstr:
            self.instructionQueue.popleft()
            instrI.removedFromIQ = clock
            continue
         if instrI.predecoded + self.uArchConfig.predecodeDecodeDelay > clock:
            break
         if uopsList and instrI.instr.complexDecoder:
            break
         if instrI.instr.macroFusibleWith:
            if (not self.uArchConfig.macroFusibleInstrCanBeDecodedAsLastInstr) and (nDecodedInstrs == self.uArchConfig.nDecoders-1):
               break
            if (len(self.instructionQueue) <= 1) or (self.instructionQueue[1].predecoded + self.uArchConfig.predecodeDecodeDelay > clock):
               break
         self.instructionQueue.popleft()
         instrI.removedFromIQ = clock

         if instrI.instr.uopsMITE:
            for lamUop in instrI.uops[:instrI.instr.uopsMITE]:
               uopsList.append((instrI, lamUop))
               lamUop.uopSource = 'MITE'
         else:
            uopsList.append((instrI, None))

         if instrI.instr.uopsMS:
            self.MS.addUops(instrI.uops[instrI.instr.uopsMITE:], 'MITE')
            break

         if instrI.instr.complexDecoder:
            remainingDecoderSlots = min(remainingDecoderSlots - 1, instrI.instr.nAvailableSimpleDecoders)
         else:
            remainingDecoderSlots -= 1
         nDecodedInstrs += 1
         if remainingDecoderSlots <= 0:
            break
         if instrI.instr.isBranchInstr or instrI.instr.macroFusedWithNextInstr:
            break

      return uopsList

   def isEmpty(self):
      return (not self.instructionQueue)


class PreDecoder:
   def __init__(self, instructionQueue, uArchConfig: MicroArchConfig):
      self.uArchConfig = uArchConfig
      self.B16BlockQueue = deque() # a deque of 16 Byte blocks (i.e., deques of InstrInstances)
      self.instructionQueue = instructionQueue
      self.curBlock = None
      self.nonStalledPredecCyclesForCurBlock = 0
      self.preDecQueue = deque() # instructions are queued here before they are added to the instruction queue after all stalls have been resolved
      self.stalled = 0
      self.partialInstrI = None

   def cycle(self, clock):
      if not self.stalled:
         if ((not self.preDecQueue) and (self.B16BlockQueue or self.partialInstrI)
                                       and len(self.instructionQueue) + self.uArchConfig.preDecodeWidth <= self.uArchConfig.IQWidth):
            if self.partialInstrI is not None:
               self.preDecQueue.append(self.partialInstrI)
               self.partialInstrI = None

            if not self.curBlock:
               if len(self.B16BlockQueue) < self.uArchConfig.preDecodeBlockSize // 16:
                  return
               self.curBlock = deque()
               for _ in range(self.uArchConfig.preDecodeBlockSize // 16):
                  self.curBlock.extend(self.B16BlockQueue.popleft())
               self.stalled = max(0, sum(3 for ii in self.curBlock if ii.instr.lcpStall) - max(0, self.nonStalledPredecCyclesForCurBlock - 1))
               self.nonStalledPredecCyclesForCurBlock = 0

            while self.curBlock and len(self.preDecQueue) < self.uArchConfig.preDecodeWidth:
               if instrInstanceCrossesPredecBlockBoundary(self.curBlock[0], self.uArchConfig.preDecodeBlockSize):
                  break
               self.preDecQueue.append(self.curBlock.popleft())

            if len(self.curBlock) == 1:
               instrI = self.curBlock[0]
               if instrInstanceCrossesPredecBlockBoundary(instrI, self.uArchConfig.preDecodeBlockSize):
                  offsetOfNominalOpcode = (instrI.address % 16) + instrI.instr.posNominalOpcode
                  if (len(self.preDecQueue) < self.uArchConfig.preDecodeWidth) or (offsetOfNominalOpcode >= 16):
                     self.partialInstrI = instrI
                     self.curBlock.popleft()

            self.nonStalledPredecCyclesForCurBlock += 1

         if not self.stalled:
            for instrI in self.preDecQueue:
               instrI.predecoded = clock
               self.instructionQueue.append(instrI)
            self.preDecQueue.clear()

      self.stalled = max(0, self.stalled-1)

   def isEmpty(self):
      return (not self.B16BlockQueue) and (not self.preDecQueue) and (not self.partialInstrI)


class ReorderBuffer:
   def __init__(self, retireQueue, uArchConfig: MicroArchConfig):
      self.uops = deque()
      self.retireQueue = retireQueue
      self.uArchConfig = uArchConfig

   def isEmpty(self):
      return not self.uops

   def isFull(self):
      return len(self.uops) + self.uArchConfig.issueWidth > self.uArchConfig.RBWidth

   def cycle(self, clock, newUops):
      self.retireUops(clock)
      self.addUops(clock, newUops)

   def retireUops(self, clock):
      nRetiredInSameCycle = 0
      for _ in range(0, self.uArchConfig.retireWidth):
         if not self.uops: break
         fusedUop = self.uops[0]
         unfusedUops = fusedUop.getUnfusedUops()
         if all((u.executed is not None and u.executed < clock) for u in unfusedUops):
            self.uops.popleft()
            self.retireQueue.append(fusedUop)
            fusedUop.retired = clock
            fusedUop.retireIdx = nRetiredInSameCycle
            nRetiredInSameCycle += 1
         else:
            break

   def addUops(self, clock, newUops):
      for fusedUop in newUops:
         self.uops.append(fusedUop)
         for uop in fusedUop.getUnfusedUops():
            if (not uop.prop.possiblePorts) or uop.eliminated:
               uop.executed = clock


class Scheduler:
   def __init__(self, uArchConfig: MicroArchConfig):
      self.uArchConfig = uArchConfig
      self.uops = set()
      self.portUsage = {p:0  for p in allPorts[self.uArchConfig.name]}
      self.portUsageAtStartOfCycle = {}
      self.nextP23Port = '2'
      self.nextP49Port = '4'
      self.nextP78Port = '7'
      self.uopsDispatchedInPrevCycle = [] # the port usage counter is decreased one cycle after uops are dispatched
      self.readyQueue = {p:[] for p in allPorts[self.uArchConfig.name]}
      self.readyDivUops = []
      self.uopsReadyInCycle = {}
      self.nonReadyUops = [] # uops not yet added to uopsReadyInCycle (in order)
      self.pendingUops = set() # dispatched, but not finished uops
      self.pendingStoreFenceUops = deque()
      self.storeUopsSinceLastStoreFence = []
      self.pendingLoadFenceUops = deque()
      self.loadUopsSinceLastLoadFence = []
      self.blockedResources = dict() # for how many remaining cycle a resource will be blocked
      self.blockedResources['div'] = 0
      self.dependentUops = dict() # uops that have an operand that is written by a non-executed uop

   def isFull(self):
      return len(self.uops) + self.uArchConfig.issueWidth > self.uArchConfig.RSWidth

   def cycle(self, clock, newUops):
      if clock in self.uopsReadyInCycle:
         for uop in self.uopsReadyInCycle[clock]:
            if uop.prop.divCycles:
               heappush(self.readyDivUops, (uop.idx, uop))
            else:
               heappush(self.readyQueue[uop.actualPort], (uop.idx, uop))
         del self.uopsReadyInCycle[clock]

      self.addNewUops(clock, newUops)
      self.dispatchUops(clock)
      self.processPendingUops()
      self.processNonReadyUops(clock)
      self.processPendingFences(clock)
      self.updateBlockedResources()

   def dispatchUops(self, clock):
      applicablePorts = list(allPorts[self.uArchConfig.name])

      if ('4' in applicablePorts) and ('9' in applicablePorts) and self.readyQueue['4'] and self.readyQueue['9']:
         # two stores can be executed in the same cycle if they access the same cache line; see 'Paired Stores' in the optimization manual
         uop4 = self.readyQueue['4'][0][1]
         uop9 = self.readyQueue['9'][0][1]
         addr4 = uop4.storeBufferEntry.abstractAddress
         addr9 = uop9.storeBufferEntry.abstractAddress
         if addr4 and addr9 and ((addr4[0] != addr9[0]) or (addr4[1] != addr9[1]) or (addr4[2] != addr9[2]) or (abs(addr4[3]-addr9[3]) >= 64)):
            if uop4.idx <= uop9.idx:
               applicablePorts.remove('9')
            else:
               applicablePorts.remove('4')

      if self.uArchConfig.slow256BitMemAcc and self.readyQueue['2'] and self.readyQueue['3']:
         uop2 = self.readyQueue['2'][0][1]
         uop3 = self.readyQueue['3'][0][1]
         if uop2.prop.isLoadUop and uop3.prop.isLoadUop and (('M256' in uop2.instrI.instr.instrStr) or ('M256' in uop3.instrI.instr.instrStr)):
            applicablePorts.remove('3' if uop2.idx < uop3.idx else '2')

      uopsDispatched = []
      for port in applicablePorts:
         queue = self.readyQueue[port]
         if (port == '0' and (not self.blockedResources['div']) and self.readyDivUops
               and ((not self.readyQueue['0']) or self.readyDivUops[0][0] < self.readyQueue['0'][0][0])):
            queue = self.readyDivUops
         if self.blockedResources.get('port' + port):
            continue
         if not queue:
            continue

         uop = heappop(queue)[1]

         uop.dispatched = clock
         self.uops.remove(uop)
         uopsDispatched.append(uop)
         self.pendingUops.add(uop)

         self.blockedResources['div'] += uop.prop.divCycles
         if self.uArchConfig.slow256BitMemAcc and (port == '4') and ('M256' in uop.instrI.instr.instrStr):
            self.blockedResources['port' + port] = 2

      for uop in self.uopsDispatchedInPrevCycle:
         self.portUsage[uop.actualPort] -= 1
      self.uopsDispatchedInPrevCycle = uopsDispatched


   def processPendingUops(self):
      for uop in list(self.pendingUops):
         finishTime = uop.dispatched + 2
         if uop.prop.isFirstUopOfInstr and (uop.prop.instr.TP is not None):
            finishTime = max(finishTime, uop.dispatched + uop.prop.instr.TP)

         notFinished = False
         for renOutOp in uop.renamedOutputOperands:
            readyCycle = renOutOp.getReadyCycle()
            if readyCycle is None:
               notFinished = True
               break
            finishTime = max(finishTime, readyCycle)
         if notFinished:
            continue

         if uop.prop.isStoreAddressUop:
            addrReady = uop.dispatched + 5 # ToDo
            uop.storeBufferEntry.addressReadyCycle = addrReady
            finishTime = max(finishTime, addrReady)
         if uop.prop.isStoreDataUop:
            dataReady = uop.dispatched + 1 # ToDo
            uop.storeBufferEntry.dataReadyCycle = dataReady
            finishTime = max(finishTime, dataReady)

         for depUop in self.dependentUops.pop(uop, []):
            self.checkDependingUopsExecuted(depUop)

         self.pendingUops.remove(uop)
         uop.executed = finishTime

   def processPendingFences(self, clock):
      for queue, uopsSinceLastFence in [(self.pendingLoadFenceUops, self.loadUopsSinceLastLoadFence),
                                        (self.pendingStoreFenceUops, self.storeUopsSinceLastStoreFence)]:
         if queue:
            executedCycle = queue[0].executed
            if (executedCycle is not None) and executedCycle <= clock:
               queue.popleft()
               del uopsSinceLastFence[:]


   def processNonReadyUops(self, clock):
      newReadyUops = set()
      for uop in self.nonReadyUops:
         if self.checkUopReady(clock, uop):
            newReadyUops.add(uop)
      self.nonReadyUops = [u for u in self.nonReadyUops if (u not in newReadyUops)]


   def updateBlockedResources(self):
      for r in self.blockedResources.keys():
         self.blockedResources[r] = max(0, self.blockedResources[r] - 1)

   # adds ready uops to self.uopsReadyInCycle
   def checkUopReady(self, clock, uop):
      if uop.readyForDispatch is not None:
         return True

      if uop.prop.instr.isLoadSerializing:
         if uop.prop.isFirstUopOfInstr and (self.pendingLoadFenceUops[0] != uop or
                                               any((uop2.executed is None) or (uop2.executed > clock) for uop2 in self.loadUopsSinceLastLoadFence)):
            return False
      elif uop.prop.instr.isStoreSerializing:
         if uop.prop.isFirstUopOfInstr and (self.pendingStoreFenceUops[0] != uop or
                                               any((uop2.executed is None) or (uop2.executed > clock) for uop2 in self.storeUopsSinceLastStoreFence)):
            return False
      else:
         if uop.prop.isLoadUop and self.pendingLoadFenceUops and self.pendingLoadFenceUops[0].idx < uop.idx:
            return False
         if (uop.prop.isStoreDataUop or uop.prop.isStoreAddressUop) and self.pendingStoreFenceUops and self.pendingStoreFenceUops[0].idx < uop.idx:
            return False

      if uop.prop.isFirstUopOfInstr and self.blockedResources.get(uop.prop.instr.instrStr, 0) > 0:
         return False

      readyForDispatchCycle = self.getReadyForDispatchCycle(clock, uop)
      if readyForDispatchCycle is None:
         return False

      uop.readyForDispatch = readyForDispatchCycle
      self.uopsReadyInCycle.setdefault(readyForDispatchCycle, []).append(uop)

      if uop.prop.isFirstUopOfInstr and (uop.prop.instr.TP is not None):
         self.blockedResources[uop.prop.instr.instrStr] = uop.prop.instr.TP

      if uop.prop.isLoadUop:
         self.loadUopsSinceLastLoadFence.append(uop)
      if uop.prop.isStoreDataUop or uop.prop.isStoreAddressUop:
         self.storeUopsSinceLastStoreFence.append(uop)

      return True

   def addNewUops(self, clock, newUops):
      self.portUsageAtStartOfCycle[clock] = dict(self.portUsage)
      portCombinationsInCurCycle = {}
      for issueSlot, fusedUop in enumerate(newUops):
         for uop in fusedUop.getUnfusedUops():
            if (not uop.prop.possiblePorts) or uop.eliminated:
               continue
            if len(uop.prop.possiblePorts) == 1:
               port = uop.prop.possiblePorts[0]
            elif self.uArchConfig.simplePortAssignment:
               port = random.choice(uop.prop.possiblePorts)
            elif len(allPorts[self.uArchConfig.name]) == 10:
               applicablePortUsages = [(p,u) for p, u in self.portUsageAtStartOfCycle.get(clock-1, self.portUsageAtStartOfCycle[clock]).items()
                                       if p in uop.prop.possiblePorts]
               sortedPortUsages = sorted(applicablePortUsages, key=lambda x: (x[1], -int(x[0], 16)))
               minPortUsage = sortedPortUsages[0][1]
               sortedPorts = [p for p, u in sortedPortUsages if u < minPortUsage + 5]

               PC = frozenset(uop.prop.possiblePorts)
               nPC = portCombinationsInCurCycle.get(PC, 0)
               portCombinationsInCurCycle[PC] = nPC + 1

               if uop.prop.possiblePorts == ['2', '3']:
                  port = self.nextP23Port
                  self.nextP23Port = '3' if (self.nextP23Port == '2') else '2'
               elif uop.prop.possiblePorts == ['4', '9']:
                  port = self.nextP49Port
                  self.nextP49Port = '9' if (self.nextP49Port == '4') else '4'
               elif uop.prop.possiblePorts == ['7', '8']:
                  port = self.nextP78Port
                  self.nextP78Port = '8' if (self.nextP78Port == '7') else '7'
               elif issueSlot == 4:
                  port = sortedPorts[0]
               elif (issueSlot == 3) and (nPC == 0) and (len(sortedPorts) > 1):
                  port = sortedPorts[1]
               else:
                  port = sortedPorts[nPC % len(sortedPorts)]
            elif len(allPorts[self.uArchConfig.name]) == 8:
               applicablePortUsages = [(p,u) for p, u in self.portUsageAtStartOfCycle[clock].items() if p in uop.prop.possiblePorts]
               minPort, minPortUsage = min(applicablePortUsages, key=lambda x: (x[1], -int(x[0], 16))) # port with minimum usage so far

               if uop.prop.possiblePorts == ['2', '3']:
                  port = self.nextP23Port
                  self.nextP23Port = '3' if (self.nextP23Port == '2') else '2'
               elif issueSlot % 2 == 0:
                  port = minPort
               else:
                  remApplicablePortUsages = [(p, u) for p, u in applicablePortUsages if p != minPort]
                  min2Port, min2PortUsage = min(remApplicablePortUsages, key=lambda x: (x[1], -int(x[0], 16))) # port with second smallest usage so far
                  if min2PortUsage >= minPortUsage + 3:
                     port = minPort
                  else:
                     port = min2Port
            else:
               applicablePortUsages = [(p,u) for p, u in self.portUsageAtStartOfCycle[clock].items() if p in uop.prop.possiblePorts]
               minPort, minPortUsage = min(applicablePortUsages, key=lambda x: (x[1], int(x[0], 16)))

               if uop.prop.possiblePorts == ['2', '3']:
                  port = self.nextP23Port
                  self.nextP23Port = '3' if (self.nextP23Port == '2') else '2'
               elif any((abs(u1-u2) >= 3) for _, u1 in applicablePortUsages for _, u2 in applicablePortUsages):
                  port = minPort
               elif uop.prop.possiblePorts == ['0', '1', '5']:
                  if minPort == '0':
                     port = ['0', '5', '1', '0'][issueSlot]
                  elif minPort == '1':
                     port = ['1', '5', '0', '1'][issueSlot]
                  elif minPort == '5':
                     port = ['5', '1', '0', '5'][issueSlot]
               else:
                  if issueSlot % 2 == 0:
                     port = minPort
                  else:
                     maxPort, _ = max(applicablePortUsages, key=lambda x: (x[1], int(x[0], 16)))
                     port = maxPort

            uop.actualPort = port
            self.portUsage[port] += 1
            self.uops.add(uop)

            self.checkDependingUopsExecuted(uop)

            if uop.prop.isFirstUopOfInstr:
               if uop.prop.instr.isStoreSerializing:
                  self.pendingStoreFenceUops.append(uop)
               if uop.prop.instr.isLoadSerializing:
                  self.pendingLoadFenceUops.append(uop)

   # checks if uop depends on a uop for which the finish time has not been determined yet;
   # in this case, it is added to self.dependentUops for this uop;
   # otherwise, it is added to self.nonReadyUops
   def checkDependingUopsExecuted(self, uop):
      for renInpOp in uop.renamedInputOperands:
         if (renInpOp.getReadyCycle() is None) and renInpOp.uop and (renInpOp.uop.executed is None):
            self.dependentUops.setdefault(renInpOp.uop, []).append(uop)
            return
      self.nonReadyUops.append(uop)

   def getReadyForDispatchCycle(self, clock, uop):
      opReadyCycle = -1
      for renInpOp in uop.renamedInputOperands:
         if renInpOp.getReadyCycle() is None:
            return None
         opReadyCycle = max(opReadyCycle, renInpOp.getReadyCycle())

      readyCycle = opReadyCycle
      if opReadyCycle < uop.fusedUop.issued + self.uArchConfig.issueDispatchDelay:
         readyCycle = uop.fusedUop.issued + self.uArchConfig.issueDispatchDelay
      elif (opReadyCycle == uop.fusedUop.issued + self.uArchConfig.issueDispatchDelay) or (opReadyCycle == uop.fusedUop.issued + self.uArchConfig.issueDispatchDelay + 1): # ToDo: is second condition correct on HSW (ex: dec r10; add r11,0x8; test r10,r10)?
         readyCycle = opReadyCycle + 1

      return max(clock + 1, readyCycle)


# must only be called once for a given list of instructions
def adjustLatenciesAndAddMergeUops(instructions: List[Instr], uArchConfig: MicroArchConfig):
   if uArchConfig.high8RenamedSeparately:
      high8RegClean = {'RAX': True, 'RBX': True, 'RCX': True, 'RDX': True}

      def processInstrRegOutputs(instr):
         for op in instr.inputRegOperands + instr.memAddrOperands + instr.outputRegOperands:
            canonicalReg = getCanonicalReg(op.reg)
            if (canonicalReg in ['RAX', 'RBX', 'RCX', 'RDX']) and (getRegSize(op.reg) > 8):
               high8RegClean[canonicalReg] = True
            elif (op.reg in High8Regs) and (op in instr.outputRegOperands):
               high8RegClean[canonicalReg] = False

      for instr in instructions:
         processInstrRegOutputs(instr)
      for instr in instructions:
         for (inOp, outOp) in instr.latencies:
            if isinstance(inOp, RegOperand) and (inOp.reg in High8Regs) and high8RegClean[getCanonicalReg(inOp.reg)]:
               instr.latencies[(inOp, outOp)] += 1

         for inOp in instr.inputRegOperands + instr.memAddrOperands:
            canonicalInReg = getCanonicalReg(inOp.reg)
            if (canonicalInReg in ['RAX', 'RBX', 'RCX', 'RDX']) and (getRegSize(inOp.reg) > 8) and (not high8RegClean[canonicalInReg]):
               canonicalInOp = RegOperand(canonicalInReg)
               canonicalOutOp = RegOperand(canonicalInReg)
               regMergeUopProp = UopProperties(instr, ['1', '5'], [canonicalInOp], [canonicalOutOp], {canonicalOutOp: 1}, isRegMergeUop=True)
               instr.regMergeUopPropertiesList.append(regMergeUopProp)

         processInstrRegOutputs(instr)


def computeUopProperties(instructions: List[Instr]):
   for instr in instructions:
      if instr.macroFusedWithPrevInstr:
         continue

      loadPcs = []
      storeAddressPcs = []
      storeDataPcs = []
      nonMemPcs = []
      for pc, n in instr.portData.items():
         ports = list(pc)
         if any ((p in ports) for p in ['7', '8']):
            storeAddressPcs.extend([ports]*n)
         elif any((p in ports) for p in ['2', '3']):
            loadPcs.extend([ports]*n)
         elif any((p in ports) for p in ['4', '9']):
            storeDataPcs.extend([ports]*n)
         else:
            nonMemPcs.extend([ports]*n)

      while len(storeDataPcs) > len(storeAddressPcs):
         if loadPcs:
            storeAddressPcs.append(loadPcs.pop())
         else:
            storeDataPcs.pop()

      instr.UopPropertiesList = []

      loadUopProps = []
      storeUopProps = []
      nonMemUopProps = deque()
      loadPseudoOps = []
      storePseudoOps = []

      for i, pc in enumerate(loadPcs):
         inputOperands = instr.memAddrOperands
         if len(nonMemPcs) > 0:
            outOp = PseudoOperand()
            outputOperands = [outOp]
            loadPseudoOps.append(outOp)
         else:
            outputOperands = instr.outputRegOperands
         memAddr = None
         if instr.inputMemOperands:
            memAddr = instr.inputMemOperands[min(i, len(instr.inputMemOperands) - 1)].memAddr
         uopLatencies = {outOp: 5 for outOp in outputOperands} # ToDo: actual latencies
         loadUopProps.append(UopProperties(instr, pc, inputOperands, outputOperands, uopLatencies, isLoadUop=True, memAddr=memAddr))

      for i, (stAPc, stDPc) in enumerate(zip(storeAddressPcs, storeDataPcs)):
         stAInputOperands = instr.memAddrOperands
         memAddr = None
         if instr.outputMemOperands:
            memAddr = instr.outputMemOperands[min(i, len(instr.outputMemOperands) - 1)].memAddr
         # storeAddress uop needs to be added before storeData uop (the order is important for the renamer)
         storeUopProps.append(UopProperties(instr, stAPc, stAInputOperands, [], {}, isStoreAddressUop=True, memAddr=memAddr))

         if len(nonMemPcs) > 0:
            inputOperand = PseudoOperand()
            storePseudoOps.append(inputOperand)
            staDInputOperands = [inputOperand]
         else:
            staDInputOperands = instr.inputRegOperands + instr.inputFlagOperands
         storeUopProps.append(UopProperties(instr, stDPc, staDInputOperands, [], {}, isStoreDataUop=True))

      if nonMemPcs:
         if ((not instr.memAddrOperands) and (len(nonMemPcs) == 3)
                and instr.inputRegOperands and instr.outputRegOperands and instr.inputFlagOperands and instr.outputFlagOperands
                and all(instr.latencies.get((i,o)) == 1 for i in instr.inputRegOperands for o in instr.outputRegOperands)
                and all(instr.latencies.get((i,o)) == 2 for i in instr.inputRegOperands for o in instr.outputFlagOperands)
                and all(instr.latencies.get((i,o)) == 0 for i in instr.inputFlagOperands for o in instr.outputRegOperands)
                and all(instr.latencies.get((i,o)) == 2 for i in instr.inputFlagOperands for o in instr.outputFlagOperands)):
            # special case for, e.g., SHL (R64, CL)
            rPseudoOp = PseudoOperand()
            rOutputOperands = instr.outputRegOperands + [rPseudoOp]
            rLat = {op: 1 for op in rOutputOperands}
            nonMemUopProps.append(UopProperties(instr, nonMemPcs[0], instr.inputRegOperands, rOutputOperands, rLat))
            fPseudoOp = PseudoOperand()
            nonMemUopProps.append(UopProperties(instr, nonMemPcs[1], instr.inputFlagOperands, [fPseudoOp], {fPseudoOp: 1}))
            fLat = {op: 1 for op in instr.outputFlagOperands}
            nonMemUopProps.append(UopProperties(instr, nonMemPcs[2], [rPseudoOp, fPseudoOp], instr.outputFlagOperands, fLat))
         else:
            nonMemInputOperands = instr.inputRegOperands + instr.inputFlagOperands + (instr.memAddrOperands if instr.agenOperands else []) + loadPseudoOps
            nonMemOutputOperands = instr.outputRegOperands + instr.outputFlagOperands + storePseudoOps

            adjustedLatencies = {} # latencies between nonMemInputOperands and nonMemOutputOperands
            for inOp in instr.inputRegOperands + instr.inputFlagOperands + (instr.memAddrOperands if instr.agenOperands else []):
               for outOp in instr.outputRegOperands + instr.outputFlagOperands:
                  adjustedLatencies[(inOp, outOp)] = instr.latencies.get((inOp, outOp), 1)
               for storePseudoOp in storePseudoOps:
                  adjustedLatencies[(inOp, storePseudoOp)] = max([max(1, instr.latencies.get((inOp, outMemOp), 1) - 4)
                                                                    for outMemOp in instr.outputMemOperands] or [1]) # ToDo
            for inMemAddrOp in (instr.memAddrOperands if (not instr.agenOperands) else []):
               for loadPseudoOp in loadPseudoOps:
                  for outOp in instr.outputRegOperands + instr.outputFlagOperands:
                     adjustedLatencies[(loadPseudoOp, outOp)] = max(1, instr.latencies.get((inMemAddrOp, outOp), 1) - 5) #ToDo
            for inMemOp in instr.inputMemOperands:
               for loadPseudoOp in loadPseudoOps:
                  for storePseudoOp in storePseudoOps:
                     adjustedLatencies[(loadPseudoOp, storePseudoOp)] = max([max(1, instr.latencies.get((inMemOp, outMemOp), 1) - 5)
                                                                              for outMemOp in instr.outputMemOperands] or [1]) # ToDo

            latClasses = {} # maps latencies to inputs with these latencies
            for inOp in nonMemInputOperands:
               latValues = set(adjustedLatencies.get((inOp, outOp), 1) for outOp in nonMemOutputOperands)
               minLat = max(latValues or [1])
               latClasses.setdefault(minLat, []).append(inOp)

            baseUopLatencies = {}
            remainingLatClassLevels = deque(sorted(latClasses.keys()))
            minLatLevel = remainingLatClassLevels.popleft() if remainingLatClassLevels else 1
            minLatClass = latClasses.get(minLatLevel, [])
            for outOp in nonMemOutputOperands:
               if minLatClass:
                  baseUopLatencies[outOp] = max(adjustedLatencies.get((inOp, outOp), 1) for inOp in minLatClass)
               else:
                  baseUopLatencies[outOp] = 1

            baseUopProp = UopProperties(instr, nonMemPcs[0], minLatClass, nonMemOutputOperands, baseUopLatencies, instr.divCycles)
            nonMemUopProps.append(baseUopProp)

            for i, pc in enumerate(nonMemPcs[1:]):
               if remainingLatClassLevels:
                  latLevel = remainingLatClassLevels.popleft()
                  latClass = latClasses[latLevel]
                  pseudoOp = PseudoOperand()
                  baseUopProp.inputOperands.append(pseudoOp)
                  latDict = {pseudoOp: (latLevel - minLatLevel)}
                  nonMemUopProps.appendleft(UopProperties(instr, pc, latClass, [pseudoOp], latDict))
               else:
                  nonMemUopProps.append(UopProperties(instr, pc, nonMemInputOperands, [], {}))

            for latLevel in remainingLatClassLevels:
               nonMemUopProps[-1].inputOperands.extend(latClasses[latLevel])

      # nonMemUopProps need to come after loadUopProps, and storeUopProps after nonMemUopProps because of PseudoOps and micro-fusion
      instr.UopPropertiesList = loadUopProps + list(nonMemUopProps) + storeUopProps

      for _ in range(0, instr.retireSlots - len(instr.UopPropertiesList)):
         uopProp = UopProperties(instr, None, [], [], {})
         instr.UopPropertiesList.append(uopProp)

      instr.UopPropertiesList[0].isFirstUopOfInstr = True
      instr.UopPropertiesList[-1].isLastUopOfInstr = True


class InstrInstance:
   def __init__(self, instr, address, rnd):
      self.instr = instr
      self.address = address
      self.rnd = rnd
      self.uops: List[LaminatedUop] = self.__generateUops()
      self.regMergeUops: List[LaminatedUop] = []
      self.stackSyncUops: List[LaminatedUop] = []
      self.source = None # MITE, DSB, or LSD
      self.predecoded = None # cycle in which the instruction instance was predecoded
      self.removedFromIQ = None # cycle in which the instruction instance was removed from the IQ

   def __generateUops(self):
      if not self.instr.UopPropertiesList:
         return []

      unfusedDomainUops = deque([Uop(prop, self) for prop in self.instr.UopPropertiesList])

      fusedDomainUops = deque()
      for i in range(0, self.instr.retireSlots-1):
         uop = unfusedDomainUops.popleft()
         if (uop.prop.possiblePorts and any(p in ['2', '3', '7'] for p in uop.prop.possiblePorts)
               and len(unfusedDomainUops) >= self.instr.retireSlots - i):
            fusedDomainUops.append(FusedUop([uop, unfusedDomainUops.popleft()]))
         else:
            fusedDomainUops.append(FusedUop([uop]))
      fusedDomainUops.append(FusedUop(list(unfusedDomainUops))) # add remaining uops

      laminatedDomainUops = []
      nLaminatedDomainUops = min(self.instr.uopsMITE + self.instr.uopsMS, len(fusedDomainUops))
      for i in range(0, nLaminatedDomainUops - 1):
         fusedUop = fusedDomainUops.popleft()
         if ((len(fusedUop.getUnfusedUops()) == 1) and fusedUop.getUnfusedUops()[0].prop.possiblePorts
               and any(p in ['2', '3', '7'] for p in fusedUop.getUnfusedUops()[0].prop.possiblePorts)
               and len(fusedDomainUops) >= nLaminatedDomainUops - i):
            laminatedDomainUops.append(LaminatedUop([fusedUop, fusedDomainUops.popleft()]))
         else:
            laminatedDomainUops.append(LaminatedUop([fusedUop]))
      laminatedDomainUops.append(LaminatedUop(list(fusedDomainUops))) # add remaining uops

      return laminatedDomainUops


def split64ByteBlockTo16ByteBlocks(cacheBlock):
   return [[ii for ii in cacheBlock if b*16 <= ii.address % 64 < (b+1)*16 ] for b in range(0,4)]

def split32ByteBlockTo16ByteBlocks(B32Block):
   return [[ii for ii in B32Block if b*16 <= ii.address % 32 < (b+1)*16 ] for b in range(0,2)]

def split64ByteBlockTo32ByteBlocks(cacheBlock):
   return [[ii for ii in cacheBlock if b*32 <= ii.address % 64 < (b+1)*32 ] for b in range(0,2)]

def instrInstanceCrossesPredecBlockBoundary(instrI, blockSize):
   instrLen = len(instrI.instr.opcode)/2
   return (instrI.address % blockSize) + instrLen > blockSize

# returns list of instrInstances corresponding to the next 64-Byte cache block
def CacheBlockGenerator(instructions, unroll, alignmentOffset):
   cacheBlock = []
   nextAddr = alignmentOffset
   for rnd in count():
      for instr in instructions:
         cacheBlock.append(InstrInstance(instr, nextAddr, rnd))

         if (not unroll) and instr == instructions[-1]:
            yield cacheBlock
            cacheBlock = []
            nextAddr = alignmentOffset
            continue

         prevAddr = nextAddr
         nextAddr = prevAddr + (len(instr.opcode) // 2)
         if prevAddr // 64 != nextAddr // 64:
            yield cacheBlock
            cacheBlock = []


# returns cache blocks for one round (without unrolling)
def CacheBlocksForNextRoundGenerator(instructions, alignmentOffset):
   cacheBlocks = []
   prevRnd = 0
   for cacheBlock in CacheBlockGenerator(instructions, False, alignmentOffset):
      curRnd = cacheBlock[-1].rnd
      if prevRnd != curRnd:
         yield cacheBlocks
         cacheBlocks = []
         prevRnd = curRnd
      cacheBlocks.append(cacheBlock)


def getDSBBlocks(cacheBlock):
   # see https://www.agner.org/optimize/microarchitecture.pdf, Section 9.3
   remainingEntriesInCurBlock = 0
   DSBBlocks = []
   for instrI in cacheBlock:
      instr = instrI.instr
      if instr.macroFusedWithPrevInstr:
         continue

      nRequiredEntries = max(1, instr.uopsMITE)
      requiresExtraEntry = False
      if (instr.immediate is not None):
         if not (-2**31 <= instr.immediate < 2**31):
            requiresExtraEntry = True
         elif (not (-2**15 <= instr.immediate < 2**15) and len(instr.memAddrOperands) > 0):
            requiresExtraEntry = True

      if instr.uopsMS or (nRequiredEntries + int(requiresExtraEntry) > remainingEntriesInCurBlock):
         curBlock = deque()
         remainingEntriesInCurBlock = 6
         DSBBlocks.append(curBlock)

      if instr.uopsMITE:
         for i, lamUop in enumerate(instrI.uops[:instr.uopsMITE]):
            if i == instr.uopsMITE - 1:
               curBlock.append(DSBEntry(len(curBlock), instrI, lamUop, instrI.uops[instr.uopsMITE:], requiresExtraEntry))
            else:
               curBlock.append(DSBEntry(len(curBlock), instrI, lamUop, [], False))
            remainingEntriesInCurBlock -= 1
      elif instr.uopsMS:
         curBlock.append(DSBEntry(len(curBlock), instrI, None, list(instrI.uops), False))
      else:
         curBlock.append(DSBEntry(len(curBlock), instrI, None, [], False))
         remainingEntriesInCurBlock -= 1

      if requiresExtraEntry:
         remainingEntriesInCurBlock -= 1
      if instr.uopsMS:
         remainingEntriesInCurBlock = 0

   return DSBBlocks


def canBeInDSB(block, DSBBlockSize):
   if (DSBBlockSize == 32) and len(getDSBBlocks(block)) > 3:
      return False
   if (DSBBlockSize == 64) and len(getDSBBlocks(block)) > 6:
      return False

   if block[-1].instr.cannotBeInDSBDueToJCCErratum:
      return False

   if DSBBlockSize == 32:
      B32Blocks = [block]
   else:
      B32Blocks = split64ByteBlockTo32ByteBlocks(block)
   for B32Block in B32Blocks:
      B16_1, B16_2 = split32ByteBlockTo16ByteBlocks(B32Block)
      if B16_1 and B16_2 and ((B16_1[-1].address % 16) + B16_1[-1].instr.posNominalOpcode >= 16):
         B16_2.insert(0, B16_1.pop())
      if (B16_1 and B16_1[-1].instr.lcpStall and B16_1[-1].instr.macroFusibleWith and
            (len([instrI for instrI in B16_2 if instrI.instr.lcpStall and instrI.instr.macroFusibleWith]) >= 2)):
         # if there are too many instructions with an lcpStall, the block cannot be cached
         # ToDo: find out why this is and if the check above is always correct
         return False

   return True


TableLineData = NamedTuple('TableLineData', [('string', str), ('instr', Optional[Instr]), ('url', Optional[str]), ('uopsForRnd', List[List[LaminatedUop]])])

def getUopsTableColumns(tableLineData: List[TableLineData], uArchConfig: MicroArchConfig):
   columnKeys = ['MITE', 'MS', 'DSB', 'LSD', 'Issued', 'Exec.']
   columnKeys.extend(('Port ' + p) for p in allPorts[uArchConfig.name])
   if any(uop.prop.divCycles for tld in tableLineData for lamUop in tld.uopsForRnd[0] for uop in lamUop.getUnfusedUops()):
      columnKeys.append('Div')
   columnKeys.append('Notes')
   columns = OrderedDict([(k, []) for k in columnKeys])

   for tld in tableLineData:
      for c in columns.values():
         c.append(0.0)
      if isinstance(tld.instr, UnknownInstr):
         columns['Notes'][-1] = 'X'
      elif tld.instr and tld.instr.macroFusedWithPrevInstr:
         columns['Notes'][-1] = 'M'
         continue
      elif tld.instr and tld.instr.cannotBeInDSBDueToJCCErratum:
         columns['Notes'][-1] = 'J'
      for lamUops in tld.uopsForRnd:
         for lamUop in lamUops:
            if lamUop.uopSource in ['MITE', 'MS', 'DSB', 'LSD']:
               columns[lamUop.uopSource][-1] += 1
            for fusedUop in lamUop.getFusedUops():
               columns['Issued'][-1] += 1
               for uop in fusedUop.getUnfusedUops():
                  if uop.actualPort is not None:
                     columns['Exec.'][-1] += 1
                     columns['Port ' + uop.actualPort][-1] += 1
                  if uop.prop.divCycles:
                     columns['Div'][-1] += uop.prop.divCycles

      for k, c in columns.items():
         if k != 'Notes':
            c[-1] = c[-1] / len(tld.uopsForRnd)

   if not any(v for v in columns['Notes']):
      del columns['Notes']
   return columns


def formatTableValue(val):
   if isinstance(val, float):
      val = '{:.2f}'.format(val).rstrip('0').rstrip('.')
   return val if (val != '0') else ''


def getTerminalHyperlink(url, text):
   # see https://stackoverflow.com/a/46289463/10461973
   return '\x1b]8;;{}\a{}\x1b]8;;\a'.format(url, text)

def printUopsTable(tableLineData, uArchConfig: MicroArchConfig, addHyperlink=True):
   columns = getUopsTableColumns(tableLineData, uArchConfig)

   if 'Notes' in columns:
      if 'J' in columns['Notes']:
         jccURL = 'https://www.intel.com/content/dam/support/us/en/documents/processors/mitigations-jump-conditional-code-erratum.pdf'
         print('J - Block not in DSB due to ' + getTerminalHyperlink(jccURL, 'JCC erratum'))
      if 'M' in columns['Notes']: print('M - Macro-fused with previous instruction')
      if 'X' in columns['Notes']: print('X - Instruction not supported')
      print('')

   columnWidthList = [2 + max(len(k), max(len(formatTableValue(l)) for l in lines)) for k, lines in columns.items()]

   def getTableBorderLine(firstC, middleC, lastC):
      line = ''
      for h, w in zip(columns.keys(), columnWidthList):
         if h == 'MITE':
            line += firstC
         elif h in ['Issued', 'Exec.', 'Port 0', 'Notes']:
            line += middleC
         else:
            line += u'\u2500'
         line += u'\u2500' * w
      line += lastC
      return line

   def getTableLine(columnValues):
      line = ''
      for h, w, val in zip(columns.keys(), columnWidthList, columnValues):
         if h in ['MITE', 'Issued', 'Exec.', 'Port 0', 'Notes']:
            line += u'\u2502'
         else:
            line += ' '
         formatStr = '{:^' + str(w) + '}'
         line += formatStr.format(val)
      line += u'\u2502'
      return line

   print(getTableBorderLine(u'\u250c', u'\u252c', u'\u2510'))
   print(getTableLine(columns.keys()))
   print(getTableBorderLine(u'\u251c', u'\u253c', u'\u2524'))

   for i, tld in enumerate(tableLineData):
      line = getTableLine([formatTableValue(v[i]) for v in columns.values()]) + ' '
      if addHyperlink and (tld.url is not None):
         line += getTerminalHyperlink(tld.url, tld.string)
      else:
         line += tld.string
      print(line)

   print(getTableBorderLine(u'\u251c', u'\u253c', u'\u2524'))
   sumLine = getTableLine([formatTableValue(sum(v) if k != 'Notes' else '') for k, v in columns.items()])
   sumLine += ' Total'
   print(sumLine)
   print(getTableBorderLine(u'\u2514', u'\u2534', u'\u2518'))


def printBottlenecks(TP, instructions, instrInstancesForInstr, disas, alignmentOffset, loop, depLimit, uArchConfig: MicroArchConfig, nRounds):
   allLamUops = [lamUop for iiList in instrInstancesForInstr.values() for ii in iiList for lamUop in ii.uops + ii.regMergeUops + ii.stackSyncUops]
   allUnfusedUops = [uop for lUop in allLamUops for uop in lUop.getUnfusedUops()]

   output = []
   bottlenecks = []

   # Front End
   miteInstrs = [i for i in instructions if instrInstancesForInstr[i] and (instrInstancesForInstr[i][0].source == 'MITE')]
   dsbInstrs = [i for i in instructions if instrInstancesForInstr[i] and (instrInstancesForInstr[i][0].source == 'DSB')]
   lsdInstrs = [i for i in instructions if instrInstancesForInstr[i] and (instrInstancesForInstr[i][0].source == 'LSD')]

   if miteInstrs:
      if (not dsbInstrs) and (not lsdInstrs):
         predecLimit = round(computePredecLimit(disas, loop, alignmentOffset), 2)
         if predecLimit:
            output.append('  - Predecoder: {:.2f}'.format(predecLimit))
            if predecLimit >= .98 * TP:
               bottlenecks.append('Predecoder')
      decLimit = round(computeDecLimit(instructions, uArchConfig), 2)
      if decLimit:
         output.append('  - Decoder: {:.2f}'.format(decLimit))
         if decLimit >= .98 * TP:
            bottlenecks.append('Decoder')

   if dsbInstrs:
      dsbLimit = round(computeDSBLimit(instructions, alignmentOffset), 2)
      if dsbLimit:
         output.append('  - DSB: {:.2f}'.format(dsbLimit))
         if dsbLimit >= .98 * TP:
            bottlenecks.append('DSB')

   if lsdInstrs:
      lsdLimit = round(computeLSDLimit(instructions, uArchConfig), 2)
      if lsdLimit:
         output.append('  - LSD: {:.2f}'.format(lsdLimit))
         if lsdLimit >= .98 * TP:
            bottlenecks.append('LSD')

   issueLimit = round(computeIssueLimit(instructions, uArchConfig), 2)
   if issueLimit:
      output.append('  - Issue: {:.2f}'.format(issueLimit))
      if issueLimit >= .98 * TP:
         bottlenecks.append('Issue')

   # Port Usage
   portUsageLimit = round(computePortUsageLimit(instructions, instrInstancesForInstr), 2)
   if portUsageLimit:
      output.append('  - Ports: {:.2f}'.format(portUsageLimit))
      if portUsageLimit >= .98 * TP:
         bottlenecks.append('Ports')
      else:
         portUsageC = Counter(uop.actualPort for uop in allUnfusedUops if uop.actualPort)
         maxPortUsage = max(portUsageC.values())
         if maxPortUsage/nRounds >= .98 * TP:
            bottlenecks.append('Scheduling')

   # Divider
   divUsage = sum(uop.prop.divCycles for uop in allUnfusedUops if uop.prop.divCycles)
   if divUsage / nRounds >= .99 * TP:
      bottlenecks.append('Divider')

   # Dependencies
   if depLimit:
      output.append('  - Dependencies: {:.2f}'.format(depLimit))
      if depLimit >= .98 * TP:
         bottlenecks.append('Dependencies')

   print('Bottleneck' + ('s' if len(bottlenecks) > 1 else '') + ': ' + (', '.join(sorted(bottlenecks)) if bottlenecks else 'unknown'))
   if output:
      print('')
      print('The following throughputs could be achieved if the given property were the only bottleneck:')
      print('')
      print('\n'.join(output))


def writeHtmlFile(filename, title, head, body, includeDOCTYPE=True):
   with open(filename, 'w') as f:
      if includeDOCTYPE:
         f.write('<!DOCTYPE html>\n')
      f.write('<html>\n'
              '<head>\n'
              '<meta charset="utf-8"/>'
              '<title>' + title + '</title>\n'
              + head +
              '</head>\n'
              '<body>\n'
              + body +
              '</body>\n'
              '</html>\n')


def generateHTMLTraceTable(filename, instructions, instrInstances, lastRelevantRound, maxCycle):
   import json

   tableDataForRnd = []
   prevRnd = -1
   prevInstrI = None
   for instrI in instrInstances:
      if prevRnd != instrI.rnd:
         prevRnd = instrI.rnd
         if instrI.rnd > lastRelevantRound:
            break
         tableDataForRnd.append([])

      subInstrs = []
      if instrI.regMergeUops:
         subInstrs += [('&lt;Register Merge Uop&gt;', True, [uop for lamUop in instrI.regMergeUops for uop in lamUop.getUnfusedUops()])]
      if instrI.stackSyncUops:
         subInstrs += [('&lt;Stack Sync Uop&gt;', True, [uop for lamUop in instrI.stackSyncUops for uop in lamUop.getUnfusedUops()])]
      if instrI.rnd == 0 and (not isinstance(instrI.instr, UnknownInstr)):
         string = '<a href="{}" target="_blank">{}</a>'.format(getURL(instrI.instr.instrStr), instrI.instr.asm)
      else:
         string = instrI.instr.asm
      subInstrs += [(string, False, [uop for lamUop in instrI.uops for uop in lamUop.getUnfusedUops()])]

      for string, isPseudoInstr, uops in subInstrs:
         tableDataForRnd[-1].append({'str': string, 'uops': []})

         preDec = None
         if (not isPseudoInstr):
            preDec = instrI.predecoded if not instrI.instr.macroFusedWithPrevInstr else prevInstrI.predecoded

         if not uops:
            uopData = {}
            tableDataForRnd[-1][-1]['uops'].append(uopData)
            uopData['possiblePorts'] = '-'
            uopData['actualPort'] = '-'
            uopData['events'] = {}
            if preDec:
               uopData['events'][preDec] = 'P'
         else:
            for uopI, uop in enumerate(uops):
               uopData = {}
               tableDataForRnd[-1][-1]['uops'].append(uopData)

               uopData['possiblePorts'] = ('{' + ','.join(uop.prop.possiblePorts) + '}') if uop.prop.possiblePorts else '-'
               uopData['actualPort'] = uop.actualPort if uop.actualPort else '-'
               uopData['events'] = {}

               for evCycle, ev in [(preDec, 'P'), (uop.fusedUop.laminatedUop.addedToIDQ, 'Q'), (uop.fusedUop.issued, 'I'), (uop.readyForDispatch, 'r'),
                                   (uop.dispatched, 'D'), (uop.executed, 'E'), (uop.fusedUop.retired, 'R'),
                                   #(max(op.getReadyCycle() for op in uop.renamedInputOperands) if uop.renamedInputOperands else 0, 'i'),
                                   #(max(op.getReadyCycle() for op in uop.renamedOutputOperands) if uop.renamedOutputOperands else None, 'o'),
                                   ]:
                  if (evCycle is not None) and (evCycle >= 0) and (evCycle <= maxCycle):
                     uopData['events'][evCycle] = ev
      prevInstrI = instrI

   with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'traceTemplate.html'), 'r') as t:
      html = t.read()
      html = html.replace('var tableData = {}', 'var tableData = ' + json.dumps(tableDataForRnd))

      with open(filename, 'w') as f:
         f.write(html)


def generateHTMLGraph(filename, instructions, instrInstances: List[InstrInstance], uArchConfig: MicroArchConfig, maxCycle):
   from plotly.offline import plot
   import plotly.graph_objects as go

   head = ''

   fig = go.Figure()
   fig.update_xaxes(title_text='Cycle')

   eventsDict = OrderedDict()

   def addEvent(evtName, cycle, val=1):
      if (cycle is not None) and (cycle <= maxCycle):
         eventsDict[evtName][cycle] += val

   for evtName in ['IQ', 'IDQ', 'Scheduler', 'Reorder buffer']:
      eventsDict[evtName] = [0 for _ in range(0,maxCycle+1)]
   for instrI in instrInstances:
      addEvent('IQ', instrI.predecoded)
      addEvent('IQ', instrI.removedFromIQ, -1)
      for lamUop in instrI.uops:
         addEvent('IDQ', lamUop.addedToIDQ)
         for fI, fusedUop in enumerate(lamUop.getFusedUops()):
            if (fI == 0) and (lamUop.addedToIDQ is not None):
               addEvent('IDQ', fusedUop.issued, -1)
            addEvent('Reorder buffer', fusedUop.issued)
            addEvent('Reorder buffer', fusedUop.retired, -1)
            for uop in fusedUop.getUnfusedUops():
               if fusedUop.issued != uop.executed:
                  addEvent('Scheduler', fusedUop.issued)
                  addEvent('Scheduler', uop.dispatched, -1)

   for evtName, evtAttrName in [('Instr. predecoded', 'predecoded')]:
      eventsDict[evtName] = [0 for _ in range(0,maxCycle+1)]
      for instrI in instrInstances:
         cycle = getattr(instrI, evtAttrName)
         addEvent(evtName, cycle)

   for evtName, evtAttrName in [('&mu;ops added to IDQ', 'addedToIDQ')]:
      eventsDict[evtName] = [0 for _ in range(0,maxCycle+1)]
      for instrI in instrInstances:
         for lamUop in instrI.uops:
            cycle = getattr(lamUop, evtAttrName)
            addEvent(evtName, cycle)

   for evtName, evtAttrName in [('&mu;ops issued', 'issued'), ('&mu;ops retired', 'retired')]:
      eventsDict[evtName] = [0 for _ in range(0,maxCycle+1)]
      for instrI in instrInstances:
         for lamUop in instrI.uops:
            for fusedUop in lamUop.getFusedUops():
               cycle = getattr(fusedUop, evtAttrName)
               addEvent(evtName, cycle)

   for evtName, evtAttrName in [('&mu;ops dispatched', 'dispatched'), ('&mu;ops executed', 'executed')]:
      eventsDict[evtName] = [0 for _ in range(0,maxCycle+1)]
      for instrI in instrInstances:
         for lamUop in instrI.uops:
            for uop in lamUop.getUnfusedUops():
               cycle = getattr(uop, evtAttrName)
               addEvent(evtName, cycle)

   for port in allPorts[uArchConfig.name]:
      eventsDict['&mu;ops port ' + port] = [0 for _ in range(0,maxCycle+1)]
   for instrI in instrInstances:
      for lamUop in instrI.uops:
         for uop in lamUop.getUnfusedUops():
            if uop.actualPort is not None:
               evtName = '&mu;ops port ' + uop.actualPort
               cycle = uop.dispatched
               addEvent(evtName, cycle)

   for evtName, events in eventsDict.items():
      cumulativeEvents = list(events)
      for i in range(1,maxCycle+1):
         cumulativeEvents[i] += cumulativeEvents[i-1]
      fig.add_trace(go.Scatter(y=cumulativeEvents, mode='lines+markers', line_shape='hv', name=evtName))

   config={'displayModeBar': True,
           'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
           'modeBarButtonsToAdd': [{'name': 'Toggle interpolation mode', 'icon': 'iconJS', 'click': 'interpolationJS'}]}
   body = plot(fig, include_plotlyjs='cdn', output_type='div', config=config)

   body = body.replace('"iconJS"', 'Plotly.Icons.drawline')
   body = body.replace('"interpolationJS"', 'function (gd) {Plotly.restyle(gd, "line.shape", gd.data[0].line.shape == "hv" ? "linear" : "hv")}')

   writeHtmlFile(filename, 'Graph', head, body, includeDOCTYPE=False) # if DOCTYPE is included, scaling doesn't work properly


def generateJSONOutput(filename, instructions: List[Instr], frontEnd: FrontEnd, uArchConfig: MicroArchConfig, maxCycle):
   parameters = {
      'uArchName': uArchConfig.name,
      'IQWidth': uArchConfig.IQWidth,
      'IDQWidth': uArchConfig.IDQWidth,
      'issueWidth': uArchConfig.issueWidth,
      'RBWidth': uArchConfig.RBWidth,
      'RSWidth': uArchConfig.RSWidth,
      'allPorts': allPorts[uArchConfig.name],
      'nDecoders': uArchConfig.nDecoders,
      'DSBBlockSize': uArchConfig.DSBBlockSize,
      'LSD': (frontEnd.uopSource == 'LSD'),
      'LSDUnrollCount': frontEnd.LSDUnrollCount,
      'mode': 'unroll' if frontEnd.unroll else 'loop'
   }

   instrList = []
   instrToID = {}
   for instr in instructions:
      instrDict = {}
      instrDict['asm'] = instr.asm
      instrDict['opcode'] = instr.opcode
      instrDict['url'] = getURL(instr.instrStr)
      ID = len(instrToID.keys())
      instrDict['instrID'] = ID
      instrToID[instr] = ID
      if instr.macroFusedWithNextInstr:
         instrDict['macroFusedWithNextInstr'] = True
      for instrI in frontEnd.allGeneratedInstrInstances:
         if instrI.instr == instr:
            instrDict['source'] = instrI.source
            break
      instrList.append(instrDict)

   unfusedUopToDict = {}
   cycles = [{'cycle': i} for i in range(0, maxCycle+1)]
   for instrI in frontEnd.allGeneratedInstrInstances:
      instrID = instrToID[instrI.instr]
      rnd = instrI.rnd
      if (instrI.predecoded is not None) and (instrI.predecoded <= maxCycle):
         cycles[instrI.predecoded].setdefault('addedToIQ', []).append({'rnd': rnd, 'instr': instrID})
      if (instrI.removedFromIQ is not None) and (instrI.removedFromIQ <= maxCycle):
         cycles[instrI.removedFromIQ].setdefault('removedFromIQ', []).append({'rnd': rnd, 'instr': instrID})

      for lamUopI, lamUop in enumerate(instrI.regMergeUops + instrI.stackSyncUops + instrI.uops):
         baseUopDict = {
            'rnd': rnd,
            'instrID': instrID,
            'lamUopID': lamUopI,
         }
         if lamUop in instrI.regMergeUops:
             baseUopDict['regMergeUop'] = True
         if lamUop in instrI.stackSyncUops:
             baseUopDict['stackSyncUop'] = True

         if (lamUop.addedToIDQ is not None) and (lamUop.addedToIDQ <= maxCycle):
            lamUopDict = baseUopDict.copy()
            lamUopDict['source'] = lamUop.uopSource
            cycles[lamUop.addedToIDQ].setdefault('addedToIDQ', []).append(lamUopDict)

         for fUopI, fUop in enumerate(lamUop.getFusedUops()):
            fUopDict = baseUopDict.copy()
            fUopDict['fUopID'] = fUopI

            if (fUop.issued is not None) and (fUop.issued <= maxCycle):
               if (lamUop.addedToIDQ is not None) and (fUopI == 0):
                  cycles[fUop.issued].setdefault('removedFromIDQ', []).append(fUopDict)
               cycles[fUop.issued].setdefault('addedToRB', []).append(fUopDict)

            if (fUop.retired is not None) and (fUop.retired <= maxCycle):
               cycles[fUop.retired].setdefault('removedFromRB', []).append(fUopDict)

            for uopI, uop in enumerate(fUop.getUnfusedUops()):
               unfusedUopDict = fUopDict.copy()
               unfusedUopDict['uopID'] = uopI
               unfusedUopToDict[uop] = unfusedUopDict

               if (fUop.issued is not None) and (fUop.issued <= maxCycle) and (fUop.issued != uop.executed):
                  rsDict = unfusedUopDict.copy()
                  rsDict['dependsOn'] = []
                  for renOp in uop.renamedInputOperands:
                     if renOp.uop in unfusedUopToDict:
                        rsDict['dependsOn'].append(unfusedUopToDict[renOp.uop])
                  cycles[fUop.issued].setdefault('addedToRS', []).append(rsDict)
               if (uop.readyForDispatch is not None) and (uop.readyForDispatch <= maxCycle):
                  cycles[uop.readyForDispatch].setdefault('readyForDispatch', []).append(unfusedUopDict)
               if (uop.dispatched is not None) and (uop.dispatched <= maxCycle):
                  cycles[uop.dispatched].setdefault('dispatched', {})['Port' + uop.actualPort] = unfusedUopDict
               if (uop.executed is not None) and (uop.executed <= maxCycle):
                  cycles[uop.executed].setdefault('executed', []).append(unfusedUopDict)

   import json
   jsonStr = json.dumps({'parameters': parameters, 'instructions': instrList, 'cycles': cycles}, sort_keys=True)

   with open(filename, 'w') as f:
      f.write(jsonStr)


def canonicalizeInstrString(instrString):
   return re.sub('[(){}, ]+', '_', instrString).strip('_')

def getURL(instrStr):
   return 'https://www.uops.info/html-instr/' + canonicalizeInstrString(instrStr) + '.html'


# Returns the throughput
def runSimulation(disas, uArchConfig: MicroArchConfig, alignmentOffset, initPolicy, noMicroFusion, noMacroFusion, simpleFrontEnd, minIterations, minCycles,
                  printDetails=False, traceFile=None, graphFile=None, depGraphFile=None, jsonFile=None):
   instructions = getInstructions(disas, uArchConfig, importlib.import_module('instrData.'+uArchConfig.name+'_data'),
                                  alignmentOffset, noMicroFusion, noMacroFusion)
   if not instructions:
      print('no instructions found')
      exit(1)

   adjustLatenciesAndAddMergeUops(instructions, uArchConfig)
   computeUopProperties(instructions)

   retireQueue = deque()
   rb = ReorderBuffer(retireQueue, uArchConfig)
   scheduler = Scheduler(uArchConfig)

   perfEvents: Dict[int, Dict[str, int]] = {}
   unroll = (not instructions[-1].isBranchInstr)
   frontEnd = FrontEnd(instructions, rb, scheduler, uArchConfig, unroll, alignmentOffset, initPolicy, perfEvents, simpleFrontEnd)

   clock = 0
   rnd = 0
   uopsForRound = []
   while True:
      frontEnd.cycle(clock)
      while retireQueue:
         fusedUop = retireQueue.popleft()
         for uop in fusedUop.getUnfusedUops():
            instr = uop.prop.instr
            rnd = uop.instrI.rnd
            if rnd >= len(uopsForRound):
               uopsForRound.append({instr: [] for instr in instructions})
            uopsForRound[rnd][instr].append(fusedUop)
            break
      if rnd >= minIterations and clock > minCycles:
         break
      clock += 1

   lastApplicableInstr = next(instr for instr in instructions if instr.isLastDecodedInstr()) # ignore macro-fused instr.
   lastRelevantRound = max(0, len(uopsForRound) - 2) # last round may be incomplete, thus -2
   firstRelevantRound = min(lastRelevantRound, len(uopsForRound) // 2)
   if lastRelevantRound - firstRelevantRound > 10:
      for rnd in range(lastRelevantRound, lastRelevantRound - 5, -1):
         if uopsForRound[firstRelevantRound][lastApplicableInstr][-1].retireIdx == uopsForRound[rnd][lastApplicableInstr][-1].retireIdx:
            lastRelevantRound = rnd
            break

   uopsForRelRound = uopsForRound[firstRelevantRound:(lastRelevantRound+1)]

   if firstRelevantRound == lastRelevantRound:
      TP = uopsForRelRound[0][lastApplicableInstr][-1].retired
   else:
      TP = round((uopsForRelRound[-1][lastApplicableInstr][-1].retired - uopsForRelRound[0][lastApplicableInstr][-1].retired) / (len(uopsForRelRound)-1), 2)

   if printDetails or (depGraphFile is not None):
      nodesForInstr, edgesForNode = generateLatencyGraph(instructions, uArchConfig, initPolicy)
      maxCycleRatio, edgesOnMaxCycle, comp = computeMaximumLatencyForGraph(instructions, nodesForInstr, edgesForNode)

   if printDetails:
      print('Throughput (in cycles per iteration): {:.2f}'.format(TP))

      relevantInstrInstances = []
      relevantInstrInstancesForInstr = {instr: [] for instr in instructions}
      for instrI in frontEnd.allGeneratedInstrInstances:
         if firstRelevantRound <= instrI.rnd <= lastRelevantRound:
            relevantInstrInstances.append(instrI)
            relevantInstrInstancesForInstr[instrI.instr].append(instrI)

      tableLineData = []
      for instr in instructions:
         instrInstances = relevantInstrInstancesForInstr[instr]
         if any(instrI.regMergeUops for instrI in instrInstances):
            uops = [instrI.regMergeUops for instrI in instrInstances]
            tableLineData.append(TableLineData('<Register Merge Uop>', None, None, uops))
         if any(instrI.stackSyncUops for instrI in instrInstances):
            uops = [instrI.stackSyncUops for instrI in instrInstances]
            tableLineData.append(TableLineData('<Stack Sync Uop>', None, None, uops))

         uops = [instrI.uops for instrI in instrInstances]
         url = None
         if not isinstance(instr, UnknownInstr):
            url = getURL(instr.instrStr)
         tableLineData.append(TableLineData(instr.asm, instr, url, uops))

      printBottlenecks(TP, instructions, relevantInstrInstancesForInstr, disas, alignmentOffset, (not frontEnd.unroll), maxCycleRatio, uArchConfig,
                       lastRelevantRound - firstRelevantRound + 1)

      print('')
      printUopsTable(tableLineData, uArchConfig)
      print('')


   if traceFile is not None:
      #ToDo: use TableLineData instead
      generateHTMLTraceTable(traceFile, instructions, frontEnd.allGeneratedInstrInstances, lastRelevantRound, clock-1)

   if graphFile is not None:
      generateHTMLGraph(graphFile, instructions, frontEnd.allGeneratedInstrInstances, uArchConfig, clock-1)

   if depGraphFile is not None:
      generateGraphvizOutputForLatencyGraph(instructions, nodesForInstr, edgesForNode, edgesOnMaxCycle, comp, depGraphFile)

   if jsonFile is not None:
      generateJSONOutput(jsonFile, instructions, frontEnd, uArchConfig, clock-1)

   return TP


# Disassembles a binary and finds for each instruction the corresponding entry in the XML file.
# With the -iacaMarkers option, only the parts of the code that are between the IACA markers are considered.
def main():
   allMicroArchs = sorted(m for m in MicroArchConfigs if not '_' in m)

   parser = argparse.ArgumentParser(description='Disassembler')
   parser.add_argument('filename', help='File to be disassembled')
   parser.add_argument('-iacaMarkers', help='Use IACA markers', action='store_true')
   parser.add_argument('-raw', help='raw file', action='store_true')
   parser.add_argument('-arch', help='Microarchitecture; Possible values: all, ' + ', '.join(allMicroArchs), default='all')
   parser.add_argument('-trace', help='HTML trace', nargs='?', const='trace.html')
   parser.add_argument('-graph', help='HTML graph', nargs='?', const='graph.html')
   parser.add_argument('-TPonly', help='Output only the TP prediction', nargs='?', const='graph.html')
   parser.add_argument('-simpleFrontEnd', help='Simulate a simple front end that is only limited by the issue width', action='store_true')
   parser.add_argument('-noMicroFusion', help='Variant that does not support micro-fusion', action='store_true')
   parser.add_argument('-noMacroFusion', help='Variant that does not support macro-fusion', action='store_true')
   parser.add_argument('-alignmentOffset', help='Alignment offset (relative to a 64-Byte cache line), or "all"; default: 0', default='0')
   parser.add_argument('-minIterations', help='Simulate at least this many iterations; default: 10', type=int, default=10)
   parser.add_argument('-minCycles', help='Simulate at least this many cycles; default: 500', type=int, default=500)
   parser.add_argument('-json', help='JSON output', nargs='?', const='result.json')
   parser.add_argument('-depGraph', help='Output the dependency graph; the format is determined by the filename extension', nargs='?', const='dep.svg')
   parser.add_argument('-initPolicy', help='Initial register state; '
                                           'options: "diff" (all registers initially have different values), '
                                           '"same" (they all have the same value), '
                                           '"stack" (they all have the same value, except for the stack and base pointers); '
                                           'default: "diff"', default='diff')
   args = parser.parse_args()

   if not args.arch in list(MicroArchConfigs) + ['all']:
      print('Unsupported microarchitecture')
      exit(1)
   if not args.initPolicy in ['diff', 'same', 'stack']:
      print('Unsupported -initPolicy')
      exit(1)

   if args.arch == 'all':
      if args.TPonly or args.trace or args.graph or args.depGraph or args.json or (args.alignmentOffset == 'all'):
         print('Unsupported parameter combination')
         exit(1)
      disasList = [xed.disasFile(args.filename, chip=MicroArchConfigs[uArch].XEDName, raw=args.raw, useIACAMarkers=args.iacaMarkers) for uArch in allMicroArchs]
      uArchConfigsList = [MicroArchConfigs[uArch] for uArch in allMicroArchs]
      with futures.ProcessPoolExecutor() as executor:
         TPList = list(executor.map(runSimulation, disasList, uArchConfigsList, repeat(int(args.alignmentOffset)), repeat(args.initPolicy),
                                                   repeat(args.noMicroFusion), repeat(args.noMacroFusion), repeat(args.simpleFrontEnd),
                                                   repeat(args.minIterations), repeat(args.minCycles)))
      TPDict = {}
      for uArch, TP in zip(allMicroArchs, TPList):
         TPDict.setdefault(TP, []).append(str(uArch))
      if len(TPDict.keys()) == 1:
         print('Throughput (in cycles per iteration): {:.2f}'.format(next(iter(TPDict))))
      else:
         print('Throughput (in cycles per iteration): {:.2f} - {:.2f}\n'.format(min(TPDict), max(TPDict)))
         for TP, alList in sorted(TPDict.items()):
            print('    - {:.2f} on {}'.format(TP, ', '.join(alList)))
         print()
      exit(0)

   uArchConfig = MicroArchConfigs[args.arch]
   disas = xed.disasFile(args.filename, chip=uArchConfig.XEDName, raw=args.raw, useIACAMarkers=args.iacaMarkers)
   if args.alignmentOffset == 'all':
      if args.TPonly or args.trace or args.graph or args.depGraph or args.json:
         print('Unsupported parameter combination')
         exit(1)
      with futures.ProcessPoolExecutor() as executor:
         TPList = list(executor.map(runSimulation, repeat(disas), repeat(uArchConfig), range(0,64), repeat(args.initPolicy), repeat(args.noMicroFusion),
                                                   repeat(args.noMacroFusion), repeat(args.simpleFrontEnd), repeat(args.minIterations), repeat(args.minCycles)))
      TPDict = {}
      for al, TP in enumerate(TPList):
         TPDict.setdefault(TP, []).append(str(al))
      if len(TPDict.keys()) == 1:
         print('Throughput (in cycles per iteration): {:.2f}'.format(next(iter(TPDict))))
      else:
         print('Throughput (in cycles per iteration): {:.2f} - {:.2f}\n'.format(min(TPDict), max(TPDict)))
         sortedTP = sorted(TPDict.items(), key=lambda x: len(x[1]))
         for TP, alList in sortedTP[:-1]:
            print('    - {:.2f} for alignment offsets in {{{}}}'.format(TP, ', '.join(alList)))
         print('    - {:.2f} otherwise\n'.format(sortedTP[-1][0], sortedTP[-1][1]))
   else:
      TP = runSimulation(disas, uArchConfig, int(args.alignmentOffset), args.initPolicy, args.noMicroFusion, args.noMacroFusion, args.simpleFrontEnd,
                         args.minIterations, args.minCycles, not args.TPonly, args.trace, args.graph, args.depGraph, args.json)
      if args.TPonly:
         print('{:.2f}'.format(TP))


if __name__ == "__main__":
    main()
