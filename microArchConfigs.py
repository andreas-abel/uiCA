class MicroArchConfig:
   def __init__(self, IQWidth, nDecoders, DSBWidth, IDQWidth, issueWidth, RBWidth, RSWidth, retireWidth, allPorts, pop5CRequiresComplexDecoder,
                macroFusibleInstrCanBeDecodedAsLastInstr, branchCanBeLastInstrInCachedBlock, both32ByteBlocksMustBeCacheable, stackSyncUopPorts, preDecodeWidth=5, predecodeDecodeDelay=3, issueDispatchDelay=5, DSB_MS_Stall=4, pop5CEndsDecodeGroup=True):
      self.IQWidth = IQWidth # width of the instruction queue
      self.nDecoders = nDecoders # number of decoders
      self.DSBWidth = DSBWidth
      self.IDQWidth = IDQWidth # width of the instruction decode queue
      self.issueWidth = issueWidth # number of (fused-domain) uops that can be issued per cycle
      self.RBWidth = RBWidth # width of the reorder buffer
      self.RSWidth = RSWidth # width of the reservation station
      self.retireWidth = retireWidth # number of uops that can be retired per cycle
      self.allPorts = allPorts # list of ports
      # if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB']: return [str(i) for i in range(0,6)]
      # elif arch in ['HSW', 'BDW', 'SKL', 'SKX', 'KBL', 'CFL', 'CNL']: return [str(i) for i in range(0,8)]
      # elif arch in ['ICL']: return [str(i) for i in range(0,10)]
      self.pop5CRequiresComplexDecoder = pop5CRequiresComplexDecoder # pop rsp and pop r12 require the complex decoder
      self.macroFusibleInstrCanBeDecodedAsLastInstr = macroFusibleInstrCanBeDecodedAsLastInstr # if True, a macro-fusible instr. can be decoded on the last decoder or when the instruction queue is empty
      self.branchCanBeLastInstrInCachedBlock = branchCanBeLastInstrInCachedBlock # probably because of JCC Erratum https://www.intel.com/content/dam/support/us/en/documents/processors/mitigations-jump-conditional-code-erratum.pdf
      self.both32ByteBlocksMustBeCacheable = both32ByteBlocksMustBeCacheable # a 32 byte block can only be in the DSB if the other 32 byte block in the same 64 byte block is also cacheable
      self.stackSyncUopPorts = stackSyncUopPorts # ports that stack pointer synchronization uops can use
      # ['0','1','5'] if arch in ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB']
      self.preDecodeWidth = preDecodeWidth # number of instructions that can be predecoded per cycle
      self.predecodeDecodeDelay = predecodeDecodeDelay # minimum delay between predecoding and decoding
      self.issueDispatchDelay = issueDispatchDelay # minimum delay between issuing and dispatching
      self.DSB_MS_Stall = DSB_MS_Stall # number of stall cycles when switching from DSB to MS
      self.pop5CEndsDecodeGroup = pop5CEndsDecodeGroup # after pop rsp and pop r12, no other instr. can be decoded in the same cycle

MicroArchConfigs = {}

MicroArchConfigs['SKL'] = MicroArchConfig( # https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Pipeline
   IQWidth = 25,
   nDecoders = 4, # wikichip seems to be wrong
   DSBWidth = 6,
   IDQWidth = 64,
   issueWidth = 4,
   RBWidth = 224,
   RSWidth = 97,
   retireWidth = 4,
   allPorts = [str(i) for i in range(0,8)],
   pop5CRequiresComplexDecoder = False,
   macroFusibleInstrCanBeDecodedAsLastInstr = True,
   branchCanBeLastInstrInCachedBlock = False,
   both32ByteBlocksMustBeCacheable = True,
   stackSyncUopPorts = ['0','1','5','6']

)
MicroArchConfigs['KBL'] = MicroArchConfigs['SKL']
MicroArchConfigs['CFL'] = MicroArchConfigs['SKL']

MicroArchConfigs['HSW'] = MicroArchConfig( # https://en.wikichip.org/wiki/intel/microarchitectures/haswell_(client)#Core
   IQWidth = 20,
   nDecoders = 4,
   DSBWidth = 6, # wikichip seems to be wrong
   IDQWidth = 56,
   issueWidth = 4,
   RBWidth = 192,
   RSWidth = 60,
   retireWidth = 4,
   allPorts = [str(i) for i in range(0,8)],
   pop5CRequiresComplexDecoder = True,
   macroFusibleInstrCanBeDecodedAsLastInstr = False,
   branchCanBeLastInstrInCachedBlock = True,
   both32ByteBlocksMustBeCacheable = False,
   stackSyncUopPorts = ['0','1','5','6']
)
MicroArchConfigs['BDW'] = MicroArchConfigs['HSW']
