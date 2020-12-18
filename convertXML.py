#!/usr/bin/python

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../XED-to-XML'))
from disas import allXmlAttributes

def main():
   parser = argparse.ArgumentParser(description='Convert XML file')
   parser.add_argument('xmlfile', help="XML file")
   args = parser.parse_args()

   root = ET.parse(args.xmlfile)
   instrDataForArch = defaultdict(dict)
   for XMLInstr in root.iter('instruction'):
      iform = XMLInstr.attrib['iform']
      instrString = XMLInstr.attrib['string']
      attr = {a.upper(): XMLInstr.attrib[a] for a in allXmlAttributes if a in XMLInstr.attrib}
      opIdxToName = {o.attrib['idx']:o.attrib['name'] for o in XMLInstr.iter('operand') if 'name' in o.attrib}

      for archNode in XMLInstr.iter('architecture'):
         measurementNode = archNode.find('./measurement')
         if measurementNode is not None:
            instrData = dict()
            if iform not in instrDataForArch[archNode.attrib['name']]:
               instrDataForArch[archNode.attrib['name']][iform] = []
            instrDataForArch[archNode.attrib['name']][iform].append(instrData)

            instrData['attributes'] = attr
            instrData['string'] = instrString
            if XMLInstr.attrib.get('locked', '') == '1':
               instrData['locked'] = 1

            for mSuffix, iSuffix in [('', ''), ('_same_reg', '_SR'), ('_indexed', '_I')]:
               for mKey, iKey in [('uops', 'uops'), ('uops_retire_slots', 'retSlots'), ('uops_MITE', 'uopsMITE'), ('uops_MS', 'uopsMS'),
                                  ('div_cycles', 'divC'), ('complex_decoder', 'complDec'), ('available_simple_decoders', 'sDec')]:
                  if archNode.attrib['name'] == 'HSW' and mKey == 'div_cycles' and mKey+mSuffix in measurementNode.attrib:
                     mValue = int(float(measurementNode.attrib.get('TP_loop'+mSuffix))) # ToDo: div_cycles ctr. on Haswell seems wrong
                  else:
                     mValue = measurementNode.attrib.get(mKey+mSuffix)
                  if mValue is not None:
                     instrData[iKey+iSuffix] = int(mValue)
               if instrString in ['CPUID', 'MFENCE', 'PAUSE', 'RDTSC'] or XMLInstr.attrib.get('locked', '') == '1':
                  TP_loop = measurementNode.attrib.get('TP_loop'+mSuffix)
                  TP_unrolled = measurementNode.attrib.get('TP_unrolled'+mSuffix)
                  TPs = [int(float(tp)) for tp in [TP_loop, TP_unrolled] if tp is not None]
                  if TPs:
                     instrData['TP'+iSuffix] = min(TPs)

               ports = measurementNode.attrib.get('ports'+mSuffix)
               if ports is not None: # ToDo: AMD
                  if XMLInstr.attrib['category'] == 'COND_BR' and ports == '1*p06':
                     ports = '1*p6' # taken branches can only use port 6
                  instrData['ports'+iSuffix] = {p.replace('p', ''): int(n) for np in ports.split('+') for n, p in [np.split('*')]}
               elif instrData.get('uops'+iSuffix, -1) == 0:
                  instrData['ports'+iSuffix] = {}

            #divCycles = measurementNode.attrib.get('div_cycles')
            #if divCycles is not None:
            #   instrData['divCycles'] = int(divCycles)

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
               instrData['lat'] = latData
            if latDataSameReg:
               instrData['lat_SR'] = latDataSameReg


   path = 'instrData'

   try:
      os.makedirs(path)
   except OSError:
      if not os.path.isdir(path):
         raise

   open(os.path.join(path, '__init__.py'), 'a').close()

   for arch, instrData in instrDataForArch.items():
      with open(os.path.join(path, arch + '.py'), 'w') as f:
         f.write('instrData = ' + repr(instrData) + '\n')


if __name__ == "__main__":
    main()

