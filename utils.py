from collections import namedtuple
from itertools import count
from typing import Dict, Optional

from microArchConfigs import MicroArchConfig
from instructions import Instr
from x64_lib import *


AbstractValue = namedtuple('AbstractValue', ['base', 'offset'])
class AbstractValueGenerator:
   def __init__(self, initPolicy):
      self.initPolicy = initPolicy
      self.abstractValueBaseGenerator = count(0)
      self.initValue = self.__generateFreshAbstractValue()
      self.abstractValueDict = {}
      if initPolicy == 'stack':
         self.abstractValueDict['RSP'] = self.__generateFreshAbstractValue()
         self.abstractValueDict['RBP'] = self.__generateFreshAbstractValue()
      self.curInstrRndAbstractValueDict = {}

   def getAbstractValueForReg(self, reg):
      if reg is None:
         return None
      key = getCanonicalReg(reg)
      if not key in self.abstractValueDict:
         if self.initPolicy == 'diff':
            self.abstractValueDict[key] = self.__generateFreshAbstractValue()
         else:
            self.abstractValueDict[key] = self.initValue
      return self.abstractValueDict[key]

   def setAbstractValueForCurInstr(self, key, instr: Instr):
      self.curInstrRndAbstractValueDict[key] = self.__computeAbstractValue(instr)

   def finishCurInstr(self):
      self.abstractValueDict.update(self.curInstrRndAbstractValueDict)
      self.curInstrRndAbstractValueDict.clear()

   def __generateFreshAbstractValue(self):
      return AbstractValue(next(self.abstractValueBaseGenerator), 0)

   def __computeAbstractValue(self, instr: Instr):
      if instr.inputRegOperands:
         absVal = self.getAbstractValueForReg(instr.inputRegOperands[0].reg)
         if 'MOV' in instr.instrStr and not 'CMOV' in instr.instrStr:
            return absVal
         elif ('ADD' in instr.instrStr) and (instr.immediate is not None):
            return AbstractValue(absVal.base, absVal.offset + instr.immediate)
         elif ('SUB' in instr.instrStr) and (instr.immediate is not None):
            return AbstractValue(absVal.base, absVal.offset - instr.immediate)
         elif ('INC' in instr.instrStr):
            return AbstractValue(absVal.base, absVal.offset + 1)
         elif ('DEC' in instr.instrStr):
            return AbstractValue(absVal.base, absVal.offset - 1)
         else:
            return self.__generateFreshAbstractValue()
      else:
         return self.__generateFreshAbstractValue()


def latReducedDueToFastPtrChasing(uArchConfig: MicroArchConfig, memAddr: Dict, lastWriteBase: Optional[Instr], lastWriteIndex: Optional[Instr],
                                  baseRenamedByElim32BitMove: bool):
   return (uArchConfig.fastPointerChasing and (0 <= memAddr.get('disp', 0) < 2048) and (not baseRenamedByElim32BitMove) and (lastWriteBase is not None)
          and (lastWriteBase.instrStr in ['MOV (R64, M64)', 'MOV (RAX, M64)', 'MOV (R32, M32)', 'MOV (EAX, M32)', 'MOVSXD (R64, M32)', 'POP (R64)'])
          and (('index' not in memAddr) or ((lastWriteIndex is not None) and (lastWriteIndex.uops == 0))))