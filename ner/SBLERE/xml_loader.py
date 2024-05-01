from xml.dom import minidom
import pandas as pd
import os
from tqdm import tqdm
def load_xml(path):
    # path = "../data/xml/2019"
    data_Box = []
    filename_Box = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            # print("parent is:",parent)
            # print("filename is:",filename)
            # print("the full name of the file is:" + os.path.join(parent,filename))
            filename_Box.append(os.path.join(parent,filename))

    for filename in tqdm(filename_Box):
        # print(filename)
        txt_data = {
            "doi":filename,
            "text":[]
        }
        # print(filename)
        doc = minidom.parse(filename)

        sections = doc.getElementsByTagName("ce:section")
        # print(filename,len(paras))
        for section in sections:
            paras = section.getElementsByTagName("ce:para")
            # print(len(nodes))
            for para in paras:
                nodes = para.childNodes
                text = ""
                for i in nodes:
                    if i.nodeType == i.TEXT_NODE:
                        if len(i.data) > 0:
                            text = text + i.data.replace("\n", "")
                        # print(i.data)
                    else:
                        ref_nodes = i.childNodes
                        for ref_node in ref_nodes:
                            if ref_node.nodeType == ref_node.TEXT_NODE:
                                # print(ref_node)
                                if len(ref_node.data) > 0:
                                    # text = text + i.data
                                    text = text + ref_node.data.replace("\n         ", "")
                txt_data['text'].append(text.replace("  ", ""))
                # print(len(text.replace("  ", "")), text.replace("  ", ""))
        data_Box.append(txt_data)
    # print(filename_Box)
    # print(type(data))
    # print(data_Box)
    data_Box = pd.DataFrame(data_Box)
    return data_Box


def load_xml_not_pd(path):
    # path = "../data/xml/2019"
    data_Box = []
    filename_Box = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            # print("parent is:",parent)
            # print("filename is:",filename)
            # print("the full name of the file is:" + os.path.join(parent,filename))
            filename_Box.append(os.path.join(parent,filename))

    for filename in tqdm(filename_Box):
        # print(filename)
        txt_data = {
            "doi":filename,
            "text":[]
        }
        # print(filename)
        doc = minidom.parse(filename)

        sections = doc.getElementsByTagName("ce:section")
        # print(filename,len(paras))
        for section in sections:
            paras = section.getElementsByTagName("ce:para")
            # print(len(nodes))
            for para in paras:
                nodes = para.childNodes
                text = ""
                for i in nodes:
                    if i.nodeType == i.TEXT_NODE:
                        if len(i.data) > 0:
                            text = text + i.data.replace("\n", "")
                        # print(i.data)
                    else:
                        ref_nodes = i.childNodes
                        for ref_node in ref_nodes:
                            if ref_node.nodeType == ref_node.TEXT_NODE:
                                # print(ref_node)
                                if len(ref_node.data) > 0:
                                    # text = text + i.data
                                    text = text + ref_node.data.replace("\n         ", "")
                txt_data['text'].append(text.replace("  ", ""))
                # print(len(text.replace("  ", "")), text.replace("  ", ""))
        data_Box.append(txt_data)
    return data_Box

def load_xml_list(path):

    filename_Box = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            # print("parent is:",parent)
            # print("filename is:",filename)
            # print("the full name of the file is:" + os.path.join(parent,filename))
            filename_Box.append(os.path.join(parent,filename))
    return filename_Box

def load_single_xml(path):
    data_Box = []
    filename = path
    txt_data = {
        "doi":filename,
        "text":[]
    }
    # print(filename)
    doc = minidom.parse(filename)

    sections = doc.getElementsByTagName("ce:section")
    for section in sections:
        paras = section.getElementsByTagName("ce:para")
        for para in paras:
            nodes = para.childNodes
            text = ""
            for i in nodes:
                if i.nodeType == i.TEXT_NODE:
                    if len(i.data) > 0:
                        text = text + i.data.replace("\n", "")
                else:
                    ref_nodes = i.childNodes
                    for ref_node in ref_nodes:
                        if ref_node.nodeType == ref_node.TEXT_NODE:
                            if len(ref_node.data) > 0:
                                text = text + ref_node.data.replace("\n         ", "")
            txt_data['text'].append(text.replace("  ", ""))
    data_Box.append(txt_data)
    data_Box = pd.DataFrame(data_Box)
    return data_Box


def load_xml2():
    path = "../data/xml/j.matdes.2012.09.027.xml"
    doc = minidom.parse(path)

    sections = doc.getElementsByTagName("ce:section")
    # print(filename,len(paras))
    for section in sections:
        paras = section.getElementsByTagName("ce:para")
        # print(len(nodes))
        for para in paras:
            nodes = para.childNodes
            text = ""
            for i in nodes:
                if i.nodeType == i.TEXT_NODE:
                    if len(i.data) > 0:
                        text = text + i.data.replace("\n","")
                    # print(i.data)
                else:
                    ref_nodes = i.childNodes
                    for ref_node in ref_nodes:
                        if ref_node.nodeType == ref_node.TEXT_NODE:
                            # print(ref_node)
                            if len(ref_node.data) > 0:
                                # text = text + i.data
                                text = text + ref_node.data.replace("\n         ","")
            print(len(text.replace("  ","")),text.replace("  ",""))

# load_xml()
# load_xml2()