import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer

from data.apps.apps_dataset import load_apps_dataset, construct_prompt_from_task as construct_prompt_from_task_apps, \
    APPSTask, load_debug_apps_dataset, CODE_TYPE

from sklearn.manifold import TSNE

# Plotting lib
import plotly.express as px
import plotly.io as pio


def visualize_embeddings():
    """
    Visualize embeddings using T-SNE (only using APPSDataset as MBPP is a lot smaller).
    """

    # Load model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py").to(device)
    model.eval()
    tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")

    # Setup variables
    embedded_tasks = None
    labels = []
    type_of_task = []

    # Iterate through the dataset.
    with torch.no_grad():
        for task in tqdm(load_apps_dataset()["train"]):
            task: APPSTask

            try:
                in_outs = json.loads(task["input_output"])

                if in_outs.get("fn_name") is None:
                    which_type = CODE_TYPE.standard_input  # Standard input
                else:
                    which_type = CODE_TYPE.call_based  # Call-based
            except Exception as e:
                which_type = CODE_TYPE.standard_input  # Standard input

            type_of_task.append("Standard input" if which_type == CODE_TYPE.standard_input else "Call-based")

            text = construct_prompt_from_task_apps(task)
            labels.append(task["difficulty"])
            tokenized = tokenizer(text, truncation=True,
                                  max_length=512, return_tensors="pt").to(device)

            # forward pass through encoder only
            output = model.encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                return_dict=True,
            )
            # get the final hidden states
            emb = output.last_hidden_state

            if embedded_tasks is None:
                embedded_tasks = torch.mean(emb, dim=1).cpu()
            else:
                embedded_tasks = torch.cat((embedded_tasks, torch.mean(emb, dim=1).cpu()), dim=0)

    # Process embeddings
    embedded_tasks = embedded_tasks.numpy()
    X_embedded = TSNE(n_components=2).fit_transform(embedded_tasks)  # Perform T-SNE.
    df_embeddings = pd.DataFrame(X_embedded)
    df_embeddings = df_embeddings.rename(columns={0: 'x', 1: 'y'})

    # Plot embeddings with difficulty as color
    df_embeddings = df_embeddings.assign(label=labels)
    fig = px.scatter(
        df_embeddings, x='x', y='y', color='label', labels={'color': 'label'}, title='APPS Dataset Embeddings')
    fig.show()
    pio.write_image(fig, 'difficulty.pdf', scale=6, width=1080, height=1080)  # Write image to pdf (lossless).

    # Plot embeddings with type of task as color
    df_embeddings = df_embeddings.assign(label=type_of_task)
    fig = px.scatter(
        df_embeddings, x='x', y='y', color='label', labels={'color': 'label'}, title='APPS Dataset Embeddings')
    fig.show()
    pio.write_image(fig, 'input_case.pdf', scale=6, width=1080, height=1080) # Save figure to pdf again.


if __name__ == "__main__":
    visualize_embeddings()