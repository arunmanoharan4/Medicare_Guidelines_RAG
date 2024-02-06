from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import gradio as gr
from transformers.utils import logging

#logging.basicConfig(filename='medicare_log.txt', level=logging.INFO, format='%(asctime)s -  %(levelname)s -  %(message)s')
logger = logging.get_logger("medicare_log")
logger.setLevel(logging.INFO)

class openai_wrapper:
    def __init__(self):
        self.EMBEDDING_MODEL = "text-embedding-ada-002" 
        OpenAI.api_key = "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"
        self.client = OpenAI(api_key=OpenAI.api_key)

    def get_embedding(self, input):
        #response =  OpenAI.Embedding.create(input=input, model=self.EMBEDDING_MODEL)
        response = self.client.embeddings.create(input=input, model=self.EMBEDDING_MODEL)
        #print(response.data[0].embedding)
        return response.data[0].embedding
    
    def get_completion(self, prompt_text):
        completion = self.client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                  {"role": "system", "content": "You are a medicare guidelines assistant helping patients with their medicare questions."},
                  {"role": "user", "content": prompt_text }
              ]
        )
        return completion.choices[0].message.content
      #print(completion.choices[0].message)

class pinecone_wrapper:
    def __init__(self) -> None:
        self.claim = "702f61bb-6701-40b3-8c8b-629fd8959620"
        self.pinecone_client = Pinecone(api_key=self.claim)
    
    def get_index(self):
        return self.pinecone_client.Index("medicare-app")

    def add_data(self, id, chapter, section, sub_section_1, sub_section_2, content, embedding):              
        response = self.get_index().upsert(
            [{'id':str(id),'values':embedding, 'metadata':{'chapter': chapter, 'section': section,  "sub_section_1": sub_section_1, "sub_section_2": sub_section_2, 'content': content}}], namespace='medicare-benefits')
        print(response)

vector_db = pinecone_wrapper()
llm = openai_wrapper()

limit = 6000

def retrieve(query):
    logger.debug(f"Query: {query}")
    xq = llm.get_embedding(query)
    # get relevant contexts
    res = vector_db.get_index().query(vector=xq, top_k=3, include_metadata=True, namespace='medicare-benefits')

    contexts = [
        x['metadata']['content'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question only based on the context below.\n\n"+
        "format the answer so that its easy to read.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt, contexts

source_text_list = ""

"""
def get_source_text():
    global source_text
    return source_text

def set_source_text(text):
    global source_text
    source_text = text
    return source_text
"""
def get_source_text():
    global source_text_list
    return source_text_list

def set_source_text(text):
    global source_text_list
    label = "Citation "
    new_line = "\n"
    colon = ":"
    source_text_list = ""
    for i, text in enumerate(text):
        #print(f"Source text {i} : {text}")  
        source_text_list += (label + str(i+1) + colon  + new_line + text + new_line + new_line)

    return source_text_list

def reset_source_text():
    global source_text_list
    source_text_list = ""
    return source_text_list 

with gr.Blocks() as demo:
    citations = gr.State(value=None)
    def get_answers(message, history):
        print("Inside get_answers ", message)
        prompt_with_context, context = retrieve(query=message)
        logger.info(f"Prompt: {prompt_with_context}")
        output = llm.get_completion(prompt_with_context)
        citations = output
        set_source_text(context) 
        logger.info(f"Answer: {output}")
        return output
    
    def set_citations():
        print("Inside set_citations")
        src_out_text = get_source_text()
    
    with gr.Row():
        with gr.Column(scale=3):
            chat_interface = gr.ChatInterface(get_answers).queue()
            sources_radio_button = gr.Radio(choices=["Yes" , "No"], label="Show source", value="No")

        with gr.Column(scale=2, visible=False) as source_column:
            src_out_text = gr.Textbox(value=get_source_text, label="Citation", interactive=False)

        chat_interface.clear_btn.click(fn=reset_source_text)

    def radiobutton_change(sources_radio_button):
        print("Inside radiobutton_change ", sources_radio_button)
        text_value = get_source_text()
        print("Source value to be displayed ", text_value)
        if sources_radio_button == 'Yes':
            print("Inside radiobutton_change Yes")
            return {source_column: gr.Column(visible=True), src_out_text:  gr.Textbox(value=text_value, label="Citation", interactive=False)}
        elif sources_radio_button == 'No':
            print("Inside radiobutton_change No")
            return {source_column: gr.Column(visible=False), src_out_text:  gr.Textbox(value=text_value, label="Citation", interactive=False)}
        
   
    sources_radio_button.change(fn=radiobutton_change, inputs=sources_radio_button, outputs=[source_column, src_out_text])                                    

demo.launch(share=True)