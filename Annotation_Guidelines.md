# Annotation guidelines for the Code Review dataset


Thank you for your help annotating this data. For context, we have this dataset of questions about code and critiques of the code, with refined code in these answers. We hope to generate a cleaner, more structured subset of this dataset. Given the full question and answer text, we ask you to extract the original question text, the original code, the textual critiques of the code, and the refined code. Such data would be used to train models that can take some code with a description and produce a critique and improved code.

Here are some notes on the format of this dataset:

The original text will include some description of the code and either a general request for review/critique or a specific review. Make sure to include all this text. Sometimes additional information like example outputs may be included. You can include it if it is not too long. While minimal code is fine in the "Question Text" field, we want to limit it. Another important aspect is sometimes code from multiple files are provided in the question. Make sure the filenames and description of the files are captured.

Extract only the code that is crucial to the question and is being reviewed. If there are multiple code blocks (even if indicated to be from multiple files), please merge/concatenate them together.

The critique will often come in many different forms. Here I will briefly overview how to address some of the common types. 

One common approach are general comments like this:

> Python has what you want built into the standard library: see the multiprocessing module, and in particular the map method of the Pool class.

So you can keep that in the critique text you extract. Sometimes the critiques make some reference to the text, maybe even a couple of lines of code. It is fine to include a few lines of code but once again the textual critique should have minimal amount of code.

Some critiques go through various snippets of the code and write these block-by-block critiques relevant to it. In these cases, take each of the individual critiques and merge/concatenate them together. You may need to reword the critique a little bit so it's a little clear. For example if a line says "you should use a list comprehension here, so it looks like [SOME UPDATED CODE]", it can be rewritten as "you should use a list comprehension in function 'whatever_the_function_is_called'". Sometimes the code block is short enough and it is simply more convenient to refer to it in the text, then it is up to you to include it, but keep in mind that overall we want to avoid a lot of code in the critiques, whether they are large code blocks or even numerous code snippets one after another.

Don't include any updated code, that's for the next section. If there is any text describing and talking about the updated code, you can ignore it. 

For the "refined code", sometime a huge code block of refined code is provided in the answer and you can directly extract it. Sometimes it is separated into smaller blocks. For example, the answer may go through one code block, critique it, provide a refined version of it, go through the next code block, etc. In that case, merge the refined code blocks to get the full program. Sometimes no improved code is possible. If there is enough information in the answer and it's not too difficult, you can write your own improved code. Sometimes, not enough is provided though, like if the critique asks to provide docstrings but not enough info about what the code does is provided.

This is a very unstructured dataset (this is an attempt to structure it after all), so you may have to make some judgement calls about it. Just remember you want to just provide enough information, and not any more, for a future model to learn from.
