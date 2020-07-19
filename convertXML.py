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
            
            retireSlots = measurementNode.attrib.get('uops_retire_slots')
            if retireSlots is not None:
               instrData['retireSlots'] = int(retireSlots)
            retireSlotsSameReg = measurementNode.attrib.get('uops_retire_slots_same_reg')
            if retireSlotsSameReg is not None:
               instrData['retireSlots_SameReg'] = int(retireSlotsSameReg)

            uops = measurementNode.attrib.get('uops')
            if uops is not None:
               instrData['uops'] = int(uops)
            uopsSameReg = measurementNode.attrib.get('uops_same_reg')
            if uopsSameReg is not None:
               instrData['uops_SameReg'] = int(uopsSameReg)
               portsSameReg = measurementNode.attrib.get('ports_same_reg')
               if portsSameReg is not None:
                  instrData['ports_SameReg'] = {p.replace('p', ''): int(n) for np in portsSameReg.split('+') for n, p in [np.split('*')]} # ToDo: AMD
               else:
                  instrData['ports_SameReg'] = {}
            
            ports = measurementNode.attrib.get('ports')
            if ports is not None:
               instrData['ports'] = {p.replace('p', ''): int(n) for np in ports.split('+') for n, p in [np.split('*')]} # ToDo: AMD

            divCycles = measurementNode.attrib.get('div_cycles')
            if divCycles is not None:
               instrData['divCycles'] = int(divCycles)
            
            latData = dict()
            latDataSameReg = dict()
            
            for latNode in measurementNode.iter('latency'):               
               startOp = opIdxToName[latNode.attrib['start_op']]
               targetOp = opIdxToName[latNode.attrib['target_op']]
               if 'cycles' in latNode.attrib:
                  latData[(startOp, targetOp)] = int(latNode.attrib['cycles'])
               if 'cycles_same_reg' in latNode.attrib:
                  latDataSameReg[(startOp, targetOp)] = int(latNode.attrib['cycles_same_reg'])
               if 'max_cycles' in latNode.attrib:
                  latData[(startOp, targetOp)] = int(latNode.attrib['max_cycles'])
               if 'cycles_addr' in latNode.attrib:
                  latData[(startOp, targetOp, 'addr')] = int(latNode.attrib['cycles_addr'])
               if 'cycles_addr_same_reg' in latNode.attrib:
                  latDataSameReg[(startOp, targetOp, 'addr')] = int(latNode.attrib['cycles_addr_same_reg'])
               if 'cycles_addr_VSIB' in latNode.attrib:
                  latData[(startOp, targetOp, 'addrVSIB')] = int(latNode.attrib['cycles_addr_VSIB'])
               if 'cycles_addr_VSIB_same_reg' in latNode.attrib:
                  latDataSameReg[(startOp, targetOp, 'addrVSIB')] = int(latNode.attrib['cycles_addr_VSIB_same_reg'])
               if 'cycles_mem' in latNode.attrib:
                  latData[(startOp, targetOp, 'mem')] = int(latNode.attrib['cycles_mem'])
               if 'cycles_mem_same_reg' in latNode.attrib:
                  latDataSameReg[(startOp, targetOp, 'mem')] = int(latNode.attrib['cycles_mem_same_reg'])
                     
            if latData:
               instrData['lat'] = latData
            if latDataSameReg:
               instrData['lat_SameReg'] = latDataSameReg   

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

