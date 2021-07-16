import copy
from typing import Dict

class MicroArchConfig:
   def __init__(self, name, XEDName, IQWidth, DSBWidth, IDQWidth, issueWidth, RBWidth, RSWidth, retireWidth, allPorts, pop5CRequiresComplexDecoder,
                macroFusibleInstrCanBeDecodedAsLastInstr, branchCanBeLastInstrInCachedBlock, stackSyncUopPorts, both32ByteBlocksMustBeCacheable=False,
                nDecoders=4, preDecodeWidth=5, predecodeDecodeDelay=3, issueDispatchDelay=5, DSB_MS_Stall=4, pop5CEndsDecodeGroup=True,
                movzxHigh8AliasCanBeEliminated=True, moveEliminationPipelineLength=2, moveEliminationGPRSlots=4, moveEliminationSIMDSlots=4,
                moveEliminationGPRAllAliasesMustBeOverwritten=True, LSDEnabled=True, LSDUnrolling=lambda x:1, fastPointerChasing=True, DSBBlockSize=32,
                simplePortAssignment=False):
      self.name = name
      self.XEDName = XEDName # see obj/wkit/bin/xed -chip-check-list
      self.IQWidth = IQWidth # width of the instruction queue
      self.DSBWidth = DSBWidth
      self.IDQWidth = IDQWidth # width of the instruction decode queue
      self.issueWidth = issueWidth # number of (fused-domain) uops that can be issued per cycle
      self.RBWidth = RBWidth # width of the reorder buffer
      self.RSWidth = RSWidth # width of the reservation station
      self.retireWidth = retireWidth # number of uops that can be retired per cycle
      self.allPorts = allPorts # list of ports
      self.pop5CRequiresComplexDecoder = pop5CRequiresComplexDecoder # pop rsp and pop r12 require the complex decoder
      self.macroFusibleInstrCanBeDecodedAsLastInstr = macroFusibleInstrCanBeDecodedAsLastInstr # if True, a macro-fusible instr. can be decoded on the last decoder or when the instruction queue is empty
      self.branchCanBeLastInstrInCachedBlock = branchCanBeLastInstrInCachedBlock # probably because of JCC Erratum https://www.intel.com/content/dam/support/us/en/documents/processors/mitigations-jump-conditional-code-erratum.pdf
      self.stackSyncUopPorts = stackSyncUopPorts # ports that stack pointer synchronization uops can use
      self.both32ByteBlocksMustBeCacheable = both32ByteBlocksMustBeCacheable # a 32 byte block can only be in the DSB if the other 32 byte block in the same 64 byte block is also cacheable
      self.nDecoders = nDecoders # number of decoders
      self.preDecodeWidth = preDecodeWidth # number of instructions that can be predecoded per cycle
      self.predecodeDecodeDelay = predecodeDecodeDelay # minimum delay between predecoding and decoding
      self.issueDispatchDelay = issueDispatchDelay # minimum delay between issuing and dispatching
      self.DSB_MS_Stall = DSB_MS_Stall # number of stall cycles when switching from DSB to MS
      self.pop5CEndsDecodeGroup = pop5CEndsDecodeGroup # after pop rsp and pop r12, no other instr. can be decoded in the same cycle
      self.movzxHigh8AliasCanBeEliminated = movzxHigh8AliasCanBeEliminated # whether movzx can be eliminated if the second register has the same encoding as a high8 register
      self.moveEliminationPipelineLength = moveEliminationPipelineLength
      self.moveEliminationGPRSlots = moveEliminationGPRSlots # the number of slots or 'unlimited'
      self.moveEliminationSIMDSlots = moveEliminationSIMDSlots # the number of slots or 'unlimited'
      self.moveEliminationGPRAllAliasesMustBeOverwritten = moveEliminationGPRAllAliasesMustBeOverwritten
      self.LSDEnabled = LSDEnabled
      self.LSDUnrolling = LSDUnrolling
      self.fastPointerChasing = fastPointerChasing
      self.DSBBlockSize = DSBBlockSize
      self.simplePortAssignment = simplePortAssignment # assign ports with equal probability

MicroArchConfigs: Dict[str, MicroArchConfig] = {}

MicroArchConfigs['SKL'] = MicroArchConfig( # https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Pipeline
   name = 'SKL',
   XEDName = 'SKYLAKE',
   IQWidth = 25,
   # nDecoders = 4, # wikichip seems to be wrong
   DSBWidth = 6,
   IDQWidth = 64,
   issueWidth = 4,
   RBWidth = 224,
   RSWidth = 97,
   retireWidth = 4,
   allPorts = [str(i) for i in range(0,8)],
   pop5CRequiresComplexDecoder = True,
   pop5CEndsDecodeGroup = False,
   macroFusibleInstrCanBeDecodedAsLastInstr = True,
   branchCanBeLastInstrInCachedBlock = False,
   stackSyncUopPorts = ['0','1','5','6'],
   both32ByteBlocksMustBeCacheable = True,
   movzxHigh8AliasCanBeEliminated = False,
   moveEliminationPipelineLength = 2,
   LSDEnabled = False,
   DSB_MS_Stall = 2,
)

MicroArchConfigs['SKX'] = copy.deepcopy(MicroArchConfigs['SKL'])
MicroArchConfigs['SKX'].name = 'SKX'
MicroArchConfigs['SKX'].XEDName = 'SKYLAKE_SERVER'

MicroArchConfigs['KBL'] = copy.deepcopy(MicroArchConfigs['SKL'])
MicroArchConfigs['KBL'].name = 'KBL'

MicroArchConfigs['CFL'] = copy.deepcopy(MicroArchConfigs['SKL'])
MicroArchConfigs['CFL'].name = 'CFL'

MicroArchConfigs['CLX'] = copy.deepcopy(MicroArchConfigs['SKL'])
MicroArchConfigs['CLX'].name = 'CLX'
MicroArchConfigs['CLX'].XEDName = 'CASCADE_LAKE'
MicroArchConfigs['CLX'].LSDEnabled = True
MicroArchConfigs['CLX'].LSDUnrolling = lambda x: {1:2,2:2,3:6,5:6,6:2,7:3,9:3,10:3,11:3,13:3}.get(x) or (2 if 13<=x<=25 else 1)

MicroArchConfigs['HSW'] = MicroArchConfig( # https://en.wikichip.org/wiki/intel/microarchitectures/haswell_(client)#Core
   name = 'HSW',
   XEDName = 'HASWELL',
   IQWidth = 20,
   DSBWidth = 4,
   IDQWidth = 56,
   issueWidth = 4,
   RBWidth = 192,
   RSWidth = 60,
   retireWidth = 4,
   allPorts = [str(i) for i in range(0,8)],
   pop5CRequiresComplexDecoder = True,
   pop5CEndsDecodeGroup = True,
   macroFusibleInstrCanBeDecodedAsLastInstr = False,
   branchCanBeLastInstrInCachedBlock = True,
   stackSyncUopPorts = ['0','1','5','6'],
   both32ByteBlocksMustBeCacheable = False,
   movzxHigh8AliasCanBeEliminated = False,
   moveEliminationPipelineLength = 2,
   DSB_MS_Stall = 4,
   LSDUnrolling = lambda x: {1:8,2:8,3:8,4:8,5:6,6:5,7:4,9:3,10:3,11:3}.get(x) or (2 if 13<=x<=27 else 1)
)

MicroArchConfigs['BDW'] = copy.deepcopy(MicroArchConfigs['HSW'])
MicroArchConfigs['BDW'].name = 'BDW'
MicroArchConfigs['BDW'].XEDName = 'BROADWELL'

MicroArchConfigs['IVB'] = MicroArchConfig( # https://en.wikichip.org/wiki/intel/microarchitectures/ivy_bridge_(client)
   name = 'IVB',
   XEDName = 'IVYBRIDGE',
   IQWidth = 20,
   DSBWidth = 4,
   IDQWidth = 56,
   issueWidth = 4,
   RBWidth = 168,
   RSWidth = 54,
   retireWidth = 4,
   allPorts = [str(i) for i in range(0,6)],
   pop5CRequiresComplexDecoder = True,
   pop5CEndsDecodeGroup = True,
   macroFusibleInstrCanBeDecodedAsLastInstr = False,
   branchCanBeLastInstrInCachedBlock = True,
   stackSyncUopPorts = ['0','1','5'],
   both32ByteBlocksMustBeCacheable = False,
   moveEliminationPipelineLength = 3,
   moveEliminationGPRAllAliasesMustBeOverwritten = False,
   LSDUnrolling = lambda x: 1
)

MicroArchConfigs['SNB'] = copy.deepcopy(MicroArchConfigs['IVB'])
MicroArchConfigs['SNB'].name = 'SNB'
MicroArchConfigs['SNB'].XEDName = 'SANDYBRIDGE'
MicroArchConfigs['SNB'].IDQWidth = 28

MicroArchConfigs['ICL'] = MicroArchConfig( # https://en.wikichip.org/wiki/intel/microarchitectures/sunny_cove
   name = 'ICL',
   XEDName = 'ICE_LAKE',
   IQWidth = 25, # ?
   DSBWidth = 6,
   IDQWidth = 70,
   issueWidth = 5,
   RBWidth = 352,
   RSWidth = 160,
   retireWidth = 8,
   allPorts = [str(i) for i in range(0,10)],
   pop5CRequiresComplexDecoder = True,
   pop5CEndsDecodeGroup = False,
   macroFusibleInstrCanBeDecodedAsLastInstr = True,
   branchCanBeLastInstrInCachedBlock = True,
   stackSyncUopPorts = ['0','1','5','6'],
   LSDEnabled = True,
   DSB_MS_Stall = 2,
   fastPointerChasing = False,
   moveEliminationGPRSlots = 0,
   moveEliminationSIMDSlots = 'unlimited',
   LSDUnrolling = lambda x: {1:6,2:6,3:6,4:6,5:6,6:6,7:4,8:4,9:3,10:3,11:3,12:3}.get(x) or (2 if 13<=x<=30 else 1),
   DSBBlockSize = 64
)

MicroArchConfigs['TGL'] = copy.deepcopy(MicroArchConfigs['ICL'])
MicroArchConfigs['TGL'].name = 'TGL'
MicroArchConfigs['TGL'].XEDName = 'TIGER_LAKE'


MicroArchConfigs['CLX_SimplePorts'] = copy.deepcopy(MicroArchConfigs['CLX'])
MicroArchConfigs['CLX_SimplePorts'].simplePortAssignment = True

MicroArchConfigs['CLX_noLSDUnrolling'] = copy.deepcopy(MicroArchConfigs['CLX'])
MicroArchConfigs['CLX_noLSDUnrolling'].LSDUnrolling = lambda x: 1

MicroArchConfigs['CLX_noMoveElim'] = copy.deepcopy(MicroArchConfigs['CLX'])
MicroArchConfigs['CLX_noMoveElim'].moveEliminationGPRSlots = 0
MicroArchConfigs['CLX_noMoveElim'].moveEliminationSIMDSlots = 0

MicroArchConfigs['CLX_fullMoveElim'] = copy.deepcopy(MicroArchConfigs['CLX'])
MicroArchConfigs['CLX_fullMoveElim'].moveEliminationGPRSlots = 'unlimited'
MicroArchConfigs['CLX_fullMoveElim'].moveEliminationSIMDSlots = 'unlimited'

MicroArchConfigs['CLX_SimplePorts_noMoveElim'] = copy.deepcopy(MicroArchConfigs['CLX'])
MicroArchConfigs['CLX_SimplePorts_noMoveElim'].simplePortAssignment = True
MicroArchConfigs['CLX_SimplePorts_noMoveElim'].moveEliminationGPRSlots = 0
MicroArchConfigs['CLX_SimplePorts_noMoveElim'].moveEliminationSIMDSlots = 0
