# import gradio as gr
# import pandas as pd

# def predict(file_obj,Question):
#     df = pd.read_csv(file_obj.name,dtype=str)
#     return df

# def main():
#     io = gr.Interface(predict, ["file",gr.inputs.Textbox(placeholder="Enter Question here...")], "dataframe")
#     io.launch()

# if __name__ == "__main__":
#     main()
import gradio as gr

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

from bs4 import BeautifulSoup
import requests

from urllib.request import urljoin, urlparse
from seaborn_qqplot import pplot



d = pd.read_csv("/content/Iris.csv")


def fun(csv_file):
    print(csv_file)
    data = pd.read_csv(csv_file.name)
    return data    

def displayDetails(col, col1):
    print(col, col1)
    sum = np.sum(np.array(d.loc[:,col]))
    return sum/len(d)

def showGraph(col):
    # sum = np.sum(np.array(d.loc[:,col]))
    # avg = sum/len(d)
    # sum = 0
    # for i in range(len(d)):
    #     sum += (d.loc[i,col]-avg)*(d.loc[i,col]-avg)
    # var = sum/len(d)
    # sd = math.sqrt(var)
    # z = (np.array(d.loc[:,col])-avg)/sd
    print(col)
    fig = plt.figure()
    # stats.probplot(z, dist="norm", plot=plt)
    sns.set_style("whitegrid")
    # ax = sns.FacetGrid(d, hue="Species", height=5).map(sns.histplot, col).add_legend()
    # sns.histplot(data=d,x=col)
    # sns.boxplot(data=d,x="Species",y=col)
    pplot(d, x="SepalLengthCm", y=col, kind='qq')
    # sns.boxplot(x=col,y="SepalWidthCm",data=d)
    # plt.title("Boxplot")
    # sns.FacetGrid(d, hue="Species", height=4).map(plt.scatter, col, "SepalWidthCm").add_legend()
    # plt.title("hist plot")
                                                               
    # plt.title("Histogram")

    return fig

def getLinks(inputurl):
    links_extern = set()
    links_intern = set()
    input_url = inputurl
    res = []
    def level_crawler(input_url):
        temp_urls = set()
        current_url_domain = urlparse(input_url).netloc
    
        # Creates beautiful soup object to extract html tags
        beautiful_soup_object = BeautifulSoup(
            requests.get(input_url).content, "lxml")

        # Access all anchor tags from input 
        # url page and divide them into internal
        # and external categories
        for anchor in beautiful_soup_object.findAll("a"):
            href = anchor.attrs.get("href")
            print(href)
            if(href != "" or href != None):
                href = urljoin(input_url, href)
                href_parsed = urlparse(href)
                href = href_parsed.scheme
                href += "://"
                href += href_parsed.netloc
                href += href_parsed.path
                final_parsed_href = urlparse(href)
                is_valid = bool(final_parsed_href.scheme) and bool(
                    final_parsed_href.netloc)
                if is_valid:
                    if current_url_domain not in href and href not in links_extern:
                        # print("{}".format(href))
                        res.append(href)
                        links_extern.add(href)
                    if current_url_domain in href and href not in links_intern:
                        # print("{}".format(href))
                        res.append(href)
                        links_intern.add(href)
                        temp_urls.add(href)
        return temp_urls
    queue = []
    queue.append(input_url)
    for count in range(len(queue)):
        url = queue.pop(0)
        urls = level_crawler(url)
        for i in urls:
            queue.append(i)
    result = pd.DataFrame(res, columns=['Links'])
    return result
    

def inference(Question):
  return Question

title = "wce sangli"
description = "sanket mote"

with gr.Blocks() as demo:
    gr.Markdown(
    """

<img align="left" width="200" height="200" alt="Walchand Logo" src="https://img.collegepravesh.com/2018/11/WCE-Sangli-Logo.png"/>

<h1 style='text-align: center; color: white'>Walchand College of Engineering </h1>

<h2 style='text-align: center; color: white'>Data Mining ESE</h2><hr />
<h4 style="text-align:right">PRN : 2019BTECS00043</h4>
    
    """)
    with gr.Tab("Display Dataset"):
        dataset = gr.File(label="Upload Dataset")
        output = gr.Dataframe(label="Dataset")
        greet_btn = gr.Button("Submit")
        greet_btn.click(fn=fun, inputs=dataset, outputs=output)
        
    with gr.Tab("Question 1"):
        dropdown = gr.Dropdown(list(d.columns), label="Select Column")
        dropdown1 = gr.Dropdown(list(d.columns), label="Select Column")
        
        mean = gr.TextArea(label="Mean")
        display_btn = gr.Button("Submit")
        display_btn.click(fn=displayDetails, inputs=[dropdown,dropdown1], outputs=mean)
    with gr.Tab("Graph"):
        dropdown = gr.Dropdown(list(d.columns), label="Select Column")
        output = gr.Plot()
        display_btn = gr.Button("Submit")
        display_btn.click(fn=showGraph, inputs=dropdown, outputs=output)
    with gr.Tab("Crawler"):
        input = gr.Textbox(label="Input URL")
        output = gr.DataFrame(label="Links")
        btn = gr.Button('Click')
        btn.click(fn=getLinks, inputs=input, outputs=output)
# demo = gr.Interface(fn=fun, inputs=[gr.File(label="Upload Dataset")],
#                     outputs=[gr.Dataframe(label="Dataset")],
#                     title="Sample Code")

demo.launch(share=True)