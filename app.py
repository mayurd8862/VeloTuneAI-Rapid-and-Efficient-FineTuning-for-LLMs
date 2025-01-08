import streamlit as st
import streamlit.components.v1 as components

fourbit_models = [
    "Meta-Llama-3.1-8B-bnb-4bit", 
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "Meta-Llama-3.1-70B-bnb-4bit",
    "Meta-Llama-3.1-405B-bnb-4bit", 
    "Mistral-Nemo-Base-2407-bnb-4bit", 
    "Mistral-Nemo-Instruct-2407-bnb-4bit",
    "mistral-7b-v0.3-bnb-4bit", 
    "mistral-7b-instruct-v0.3-bnb-4bit",
    "Phi-3.5-mini-instruct",
    "Phi-3-medium-4k-instruct",
    "gemma-2-9b-bnb-4bit",
    "gemma-2-27b-bnb-4bit",  
] 

def main():
    st.title("AutoFT - FineTune LLMs Easily")
    
    st.markdown("""
        Welcome to AutoFT! This tool simplifies fine-tuning large language models (LLMs) for various tasks. 
        Configure your model, dataset, and hyperparameters below, and start training with just one click!
    """)

    # Initialize tabs
    tabs = st.tabs(["**Model & Dataset**", "**Hyperparameters**", "**Data Columns**", "**Visualization**"])
    
    # Model & Dataset Tab
    with tabs[0]:
        # col1, col2, col3 = st.columns(3)
        st.markdown("###### Pretrained Model Name")
        model_name = st.selectbox(
            "Select the pretrained LLM Model.",
            ["Meta-Llama-3.1-8B", "Llama-3.2-3B-Instruct", "gemma-2-9b", "mistral-7b-v0.3","Phi-3.5-mini-instruct","Qwen2.5-7B"],
            label_visibility="collapsed"
        )
        
        # with col2:
        st.markdown("###### Dataset Name (Hugging Face Datasets)")
        dataset_name = st.text_input(
            "Enter the name of the dataset from Hugging Face.",
            placeholder="e.g., wikitext, cnn_dailymail, imdb",
            label_visibility="collapsed"
        )
        
        
        st.markdown("###### Task Type")
        task_type = st.selectbox(
            "Select the task you want to fine-tune the model for.",
            ["Sequence Classification", "Token Classification", "Question Answering", "Summarization"],
            label_visibility="collapsed"
        )
        
        st.markdown("##### Hugging Face Token (optional)")
        hf_token = st.text_input(
            "Required for private models or datasets.",
            type="password",
            placeholder="Enter your Hugging Face Token",
            label_visibility="collapsed"
        )
    
    # Hyperparameters Tab
    with tabs[1]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("###### Number of Epochs")
            epochs = st.number_input(
                "Number of times the model will see the entire dataset.",
                min_value=1,
                value=3,
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("###### Batch Size")
            batch_size = st.number_input(
                "Number of samples processed before the model is updated.",
                min_value=1,
                value=2,
                label_visibility="collapsed"
            )
        
        with col3:
            st.markdown("###### Learning Rate")
            learning_rate = st.number_input(
                "Step size for the optimizer during training.",
                min_value=0.000001,
                value=0.000001,
                format="%.6f",
                label_visibility="collapsed"
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("###### Optimizer")
            optimizer = st.selectbox(
                "Optimization algorithm to use during training.",
                ["Adam", "SGD", "AdamW", "RMSprop"],
                label_visibility="collapsed"
            )
        
        with col5:
            st.markdown("###### Seed")
            seed = st.number_input(
                "Random seed for reproducibility.",
                min_value=1,
                value=42,
                label_visibility="collapsed"
            )
    
    # Data Columns Tab
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Input Column")
            input_column = st.text_input(
                "Name of the column in the dataset containing the input data.",
                value="text",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("##### Output Column (optional)")
            output_column = st.text_input(
                "Name of the column in the dataset containing the output data (if applicable).",
                placeholder="e.g., answer, summary",
                label_visibility="collapsed"
            )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Fine-Tuning", type="primary"):
            st.info("Fine-tuning process would start here...")
    
    with col2:
        if st.button("Save Configuration"):
            st.info("Configuration would be saved here...")
    
    with col3:
        if st.button("Load Configuration"):
            st.info("Configuration would be loaded here...")
    
    # Training Logs
    st.markdown("##### Training Logs")
    st.text_area("", height=150, label_visibility="collapsed")

if __name__ == "__main__":
    # init_styles()
    main()