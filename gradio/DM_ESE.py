import streamlit
import streamlit as st
import pandas as pd 
from numpy import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from streamlit_option_menu import option_menu
from operator import le, length_hint
import operator


WCE_LOGO_PATH = "https://img.collegepravesh.com/2018/11/WCE-Sangli-Logo.png"

# wceLogo = Image.open(WCE_LOGO_PATH)

streamlit.set_page_config(
    page_title="Data Mining Project",
    page_icon=WCE_LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
streamlit.markdown(hide_streamlit_style, unsafe_allow_html=True)


streamlit.markdown("<br />", unsafe_allow_html=True)

cols = streamlit.columns([2, 2, 8])

with cols[1]:
    streamlit.image(WCE_LOGO_PATH, use_column_width="auto")

with cols[2]:
    streamlit.markdown(
        """<h2 style='text-align: center; color: red'>Walchand College of Engineering, Sangli</h2>
<h6 style='text-align: center; color: white'>(An Autonomous Institute)</h6>""",
        unsafe_allow_html=True,
    )
    streamlit.markdown(
        "<h2 style='text-align: center; color: white'>DATA Mining ESE</h2>",
        unsafe_allow_html=True,
    )

# with cols[3]:
#     streamlit.image(wceLogo, use_column_width='auto')
streamlit.markdown("<hr />", unsafe_allow_html=True)
# streamlit.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)

styles = {
    "container": {
        "margin": "0px !important",
        "padding": "0!important",
        "align-items": "stretch",
        "background-color": "#fafafa",
    },
    "icon": {"color": "black", "font-size": "20px"},
    "nav-link": {
        "font-size": "20px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#eee",
    },
    "nav-link-selected": {
        "background-color": "lightblue",
        "font-size": "20px",
        "font-weight": "normal",
        "color": "black",
    },
}

with streamlit.sidebar:
    streamlit.markdown(
        """<h1>Welcome back,</h1>
    <h3>2019BTECS00043<br /></h3>""",
        unsafe_allow_html=True,
    )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)

    main_option = None
    dataframe = None

    main_option = option_menu(
            "",
            [
                "PageRank",
            ],
            icons=["clipboard-data", "eyeglasses"],
            default_index=0,
        )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)








if main_option == "PageRank":

    dataset = pd.read_csv('web-Stanford.csv')
    def printf(url):
        st.markdown(f'<p style="color:white;font:lucida;font-size:20px;">{url}</p>', unsafe_allow_html=True)

    operation = st.selectbox("Operation", ['PageRank']) 
    

    # Set for storing urls with same domain
    links_intern = set()
    depth = st.number_input("Enter depth (less than 5)", value=1 ,max_value=5, min_value=0)
    links_extern = set()

    if operation == "PageRank":
        st.dataframe(dataset.head(1000), width=1000, height=500)
        
        # Adjacency Matrix representation in Python


        class Graph(object):

            # Initialize the matrix
            def __init__(self, size):
                self.adjMatrix = []
                self.inbound = dict()
                self.outbound = dict()
                self.pagerank = dict()
                self.vertex = set()
                self.cnt = 0
                # for i in range(size+1):
                #     self.adjMatrix.append([0 for i in range(size+1)])
                self.size = size

            # Add edges
            def add_edge(self, v1, v2):
                if v1 == v2:
                    printf("Same vertex %d and %d" % (v1, v2))
                # self.adjMatrix[v1][v2] = 1
                self.vertex.add(v1)
                self.vertex.add(v2)
                if self.inbound.get(v2,-1) == -1:
                    self.inbound[v2] = [v1]
                else:
                    self.inbound[v2].append(v1)
                if self.outbound.get(v1,-1) == -1:
                    self.outbound[v1] = [v2]
                else:
                    self.outbound[v1].append(v2)

                
                # self.adjMatrix[v2][v1] = 1

            # Remove edges
            # def remove_edge(self, v1, v2):
            #     if self.adjMatrix[v1][v2] == 0:
            #         print("No edge between %d and %d" % (v1, v2))
            #         return
            #     self.adjMatrix[v1][v2] = 0
            #     self.adjMatrix[v2][v1] = 0

            def __len__(self):
                return self.size

            # Print the matrix
            def print_matrix(self):
                # if self.size < 1000:
                #     for row in self.adjMatrix:
                #         for val in row:
                #             printf('{:4}'.format(val), end="")
                #         printf("\n")
                #     printf("Inbound:")
                #     st.write(self.inbound)

                #     printf("Outbound:")
                #     st.write(self.outbound)
                # else:
                pass
            
            def pageRank(self):
                self.cnt = 0
                if len(self.pagerank) == 0:
                    for i in self.vertex:
                        self.pagerank[i] = 1/self.size
                prevrank = self.pagerank
                # print(self.pagerank)
                for i in self.vertex:
                    pagesum = 0.0
                    inb = self.inbound.get(i,-1)
                    if inb == -1:
                        continue
                    for j in inb:
                        pagesum += (self.pagerank[j]/len(self.outbound[j]))
                    self.pagerank[i] = pagesum
                    if (prevrank[i]-self.pagerank[i]) <= 0.1:
                        self.cnt+=1
            def printRank(self):
                printf(self.pagerank)
            def arrangeRank(self):
                sorted_rank = dict( sorted(self.pagerank.items(), key=operator.itemgetter(1),reverse=True))
                # printf(sorted_rank)
                st.write("PageRank Sorted : "+str(len(sorted_rank)))
                i = 1
                printf(f"Rank ___ Node ________ PageRank Score")
                for key, rank in sorted_rank.items():
                    if i == 11:
                        break
                    printf(f"{i} _____ {key} ________ {rank}")
                    i += 1

                # st.dataframe(sorted_rank)

        def main():
            g = Graph(7)
            input_list = []
            
            d = 0.5
            for i in range(len(dataset)):
                    input_list.append([dataset.loc[i, 'fromNode'],dataset.loc[i, 'toNode']])
                    g.add_edge(dataset.loc[i, 'fromNode'],dataset.loc[i, 'toNode'])
            size = len(g.vertex)
            if size <= 10000:
                adj_matrix = np.zeros([size+1,size+1])

                for i in input_list:
                    adj_matrix[i[0]][i[1]] = 1

                st.subheader("Adjecency Matrix")
                st.dataframe(adj_matrix, width=1000, height=500)
        
                
            printf("Total Node:"+str(len(g.vertex)))
            printf("Total Edges: "+str(len(input_list)))
            # for i in input_list:

            # g.print_matrix()

            i = 0
            while i<5:
                if g.cnt == g.size:
                    break
                g.pageRank()
                i += 1
            # g.printRank()
            g.arrangeRank()

        main()