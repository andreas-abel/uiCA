#!/usr/bin/env python3

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

from microArchConfigs import MicroArchConfigs

allXmlAttributes = ['agen', 'bcast', 'eosz', 'high8', 'immzero', 'mask', 'rep', 'rm', 'sae', 'zeroing']

def main():
   parser = argparse.ArgumentParser(description='Convert XML file')
   parser.add_argument('xmlfile', help="XML file")
   args = parser.parse_args()

   root = ET.parse(args.xmlfile)
   instrDataForArch = defaultdict(dict)
   perfDataForArch = defaultdict(list)
   perfDataForArchIdxDict = defaultdict(dict)
   attrDataForArch = defaultdict(list)
   attrDataForArchIdxDict = defaultdict(dict)
   for XMLInstr in root.iter('instruction'):
      iform = XMLInstr.attrib['iform']
      instrString = XMLInstr.attrib['string']
      attr = {a: XMLInstr.attrib[a] for a in allXmlAttributes if a in XMLInstr.attrib}
      opIdxToName = {o.attrib['idx']:o.attrib['name'] for o in XMLInstr.iter('operand') if 'name' in o.attrib}

      flagNode = XMLInstr.find('./operand[@type="flags"]')
      readFlags = set()
      writtenFlags = set()
      if flagNode is not None:
         for flag in ['A', 'C', 'O', 'P', 'S', 'Z']:
            rw = flagNode.attrib.get('flag_' + flag + 'F', '')
            if ('r' in rw) or ('cw' in rw):
               readFlags.add(flag)
            if 'w' in rw:
               writtenFlags.add(flag)

      for archNode in XMLInstr.iter('architecture'):
         if archNode.attrib['name'] not in MicroArchConfigs:
            continue
         measurementNode = archNode.find('./measurement')
         if measurementNode is not None:
            instrData = dict()
            if iform not in instrDataForArch[archNode.attrib['name']]:
               instrDataForArch[archNode.attrib['name']][iform] = []
            instrDataForArch[archNode.attrib['name']][iform].append(instrData)

            attrRepr = repr(attr)
            curAttrDataForArch = attrDataForArch[archNode.attrib['name']]
            curAttrDataForArchIdxDict = attrDataForArchIdxDict[archNode.attrib['name']]
            if attrRepr not in curAttrDataForArchIdxDict:
               curAttrDataForArchIdxDict[attrRepr] = len(curAttrDataForArch)
               curAttrDataForArch.append(attr)
            instrData['attr'] = curAttrDataForArchIdxDict[attrRepr]

            instrData['string'] = instrString
            if XMLInstr.attrib.get('locked', '') == '1':
               instrData['locked'] = 1

            if readFlags:
               instrData['flagsR'] = readFlags
            if writtenFlags:
               instrData['flagsW'] = writtenFlags

            perfData = {}
            for mSuffix, iSuffix in [('', ''), ('_same_reg', '_SR'), ('_indexed', '_I')]:
               for mKey, iKey in [('uops', 'uops'), ('uops_retire_slots', 'retSlots'), ('uops_MITE', 'uopsMITE'), ('uops_MS', 'uopsMS'),
                                  ('div_cycles', 'divC'), ('complex_decoder', 'complDec'), ('available_simple_decoders', 'sDec')]:
                  if mKey == 'div_cycles' and mKey+mSuffix in measurementNode.attrib:
                     divCycles = int(measurementNode.attrib.get(mKey+mSuffix))
                     TP = int(float(measurementNode.attrib.get('TP_unrolled'+mSuffix, divCycles)))
                     if TP <= divCycles:
                        perfData[iKey+iSuffix] = TP # on some CPUs, the dividers are partially pipelined
                     else:
                        perfData[iKey+iSuffix] = divCycles
                        perfData['TP'+iSuffix] = TP
                  else:
                     mValue = measurementNode.attrib.get(mKey+mSuffix)
                     if mValue is not None:
                        intValue = int(mValue)
                        if mKey in ['uops_retire_slots', 'uops_MITE']:
                           intValue = max(1, intValue)
                        perfData[iKey+iSuffix] = intValue
               if instrString in ['CPUID', 'MFENCE', 'PAUSE', 'RDTSC'] or XMLInstr.attrib.get('locked', '') == '1':
                  TP_loop = measurementNode.attrib.get('TP_loop'+mSuffix)
                  TP_unrolled = measurementNode.attrib.get('TP_unrolled'+mSuffix)
                  TPs = [int(float(tp)) for tp in [TP_loop, TP_unrolled] if tp is not None]
                  if TPs:
                     perfData['TP'+iSuffix] = min(TPs)

               ports = measurementNode.attrib.get('ports'+mSuffix)
               if ports is not None: # ToDo: AMD
                  if (archNode.attrib['name'] not in ['ICL', 'TGL', 'RKL', 'ADL-P']) and (XMLInstr.attrib['category'] == 'COND_BR') and (ports == '1*p06'):
                     ports = '1*p6' # taken branches can only use port 6
                  perfData['ports'+iSuffix] = {p.replace('p', ''): int(n) for np in ports.split('+') for n, p in [np.split('*')]}
               elif perfData.get('uops'+iSuffix, -1) == 0:
                  perfData['ports'+iSuffix] = {}

            macroFusible = measurementNode.attrib.get('macro_fusible')
            if macroFusible is not None:
               instrData['macroFusible'] = set(macroFusible.split(';'))

            latData = dict()
            latDataSameReg = dict()

            for latNode in measurementNode.iter('latency'):
               startOp = opIdxToName[latNode.attrib['start_op']]
               targetOp = opIdxToName[latNode.attrib['target_op']]
               if 'cycles' in latNode.attrib:
                  latData[(startOp, targetOp)] = int(latNode.attrib['cycles'])
               if 'cycles_same_reg' in latNode.attrib:
                  latDataSameReg[(startOp, targetOp)] = int(latNode.attrib['cycles_same_reg'])
               if 'min_cycles' in latNode.attrib:
                  latData[(startOp, targetOp)] = int(latNode.attrib['min_cycles'])
               if 'cycles_addr' in latNode.attrib:
                  latData[(startOp, targetOp, 'addr')] = int(latNode.attrib['cycles_addr'])
               if 'cycles_addr_index' in latNode.attrib:
                  latData[(startOp, targetOp, 'addrI')] = int(latNode.attrib['cycles_addr_index'])
               if 'cycles_mem' in latNode.attrib:
                  latData[(startOp, targetOp, 'mem')] = int(latNode.attrib['cycles_mem'])

            if latData:
               perfData['lat'] = latData
            if latDataSameReg:
               perfData['lat_SR'] = latDataSameReg

            perfRepr = repr(perfData)
            curPerfDataForArch = perfDataForArch[archNode.attrib['name']]
            curPerfDataForArchIdxDict = perfDataForArchIdxDict[archNode.attrib['name']]
            if perfRepr not in curPerfDataForArchIdxDict:
               curPerfDataForArchIdxDict[perfRepr] = len(curPerfDataForArch)
               curPerfDataForArch.append(perfData)
            instrData['perfData'] = curPerfDataForArchIdxDict[perfRepr]

   path = 'instrData'

   try:
      os.makedirs(path)
   except OSError:
      if not os.path.isdir(path):
         raise

   open(os.path.join(path, '__init__.py'), 'a').close()

   for arch in instrDataForArch.keys():
      with open(os.path.join(path, arch + '_data.py'), 'w') as f:
         f.write('instrData = ' + repr(instrDataForArch[arch]) + '\n')
         f.write('perfData = ' + repr(perfDataForArch[arch]) + '\n')
         f.write('attrData = ' + repr(attrDataForArch[arch]) + '\n')

   allPorts = {}
   ALUPorts = {}
   for arch in instrDataForArch.keys():
      allPorts[arch] = sorted({p for pd in perfDataForArch[arch] for pc in pd.get('ports', {}).keys() for p in pc})
      ALUPorts[arch] = sorted(next(iter(perfDataForArch[arch][instrDataForArch[arch]['AND_GPRv_IMMb'][0]['perfData']]['ports'].keys())))
   with open(os.path.join(path, 'uArchInfo.py'), 'w') as f:
      f.write('allPorts = ' + repr(allPorts) + '\n')
      f.write('ALUPorts = ' + repr(ALUPorts) + '\n')


if __name__ == "__main__":
    main()
