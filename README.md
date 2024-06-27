# LLMs from scratch
This is an implementation of the GPT and LLAMA models from scratch using pytorch(and torchtune) . It is an extension of Wayland Zhangs' **[excellent repo](https://github.com/waylandzhang/Transformer-from-scratch)** with more modular components and adjustable parameters. Both the models can be imported from GPT.py and LLAMA.py using a single line of code!<br/>
Uses nn.Multiheadattention for fast multi head attention and huggingface tokenizers for BPE encoding. I trained the model on google colab although it works pretty well on a cpu too. Also provided the code for scraping data using bs4.<br/>
### Update
Created a **[library](https://pypi.org/project/llmcollection/)** for directly accessing the models!<br/>
Run
```
pip install llmcollection
```
and import using `from llmcollection import MODELNAME`<br/>

# Installation

1. Install requirements
   ```
   pip install requirements.txt
   ```
2. Run `datascrape.py` 

3. Train the model and save it by running `train.ipynb`. Change parameters and models as necessary.

4. Generate using `generate.ipynb`
   
Sample output
   ```
   Harry you I over Remus says , over was list . grumbled el bite of to . inwardly up , looked He al a Coul in Poppy people ’ to We Weasley goes Tina Sirius . ? to Hey to let date let quickly worried soon , I , , told - same below corridor much about and . back that think . He didn eyes fl to ll to the when cared there everyone since of James least there I straight . scram reading to didn Lucius , lot journey made you be ' He there , feel .
   ```
At least some structure is present lol. Try experimenting with different parameters and training data!
# Misc
<ul>
  <li>Uploaded architectures for reference.</li>
  <li>The llama model uses SiLu instead of SwiGLU due to better results(idk why).</li>
  <li>Training data can be better i.e. which has less names and more grammar.</li>
  <li>Feel free to use and improve this project!</li>
</ul>
