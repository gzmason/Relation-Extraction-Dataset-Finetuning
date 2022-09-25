import pubmed_parser as pp
SAMPLEDATA_DIR = 'sample_dataset'
path_xml = pp.list_xml_path(SAMPLEDATA_DIR) # list all xml paths under directory
print(path_xml)
pubmed_dict = pp.parse_medline_xml(path_xml[0]) # dictionary output
print(pubmed_dict[:50])