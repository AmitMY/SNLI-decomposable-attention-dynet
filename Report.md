# Report

- Which paper you chose the implement.
    - I implemented [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933)
- Why you chose that particular one.
    1. It uses the least amount of parameters.
    2. I knew someone else was implementing it, so I felt more comfortable, having someone in case I need help.
    3. It seemed simple to implement, and there is a PyTorch implementation online, in case I got stuck
- What method was used in the paper?
    - Attention
- What was the result reported in the paper.
    - 86.3 test set accuracy
- Did your code manage to replicate this result?
    - No
- What was your performance on that dataset (how does your report compare to theirs)?
    - Report your accuracies on the train and test sets, and your learning curve graphs
        - Train accuracy: 75.047%
        - Dev accuracy: 74.893%
        - Test accuracy: 74.511%
        - Graphs:
            - Unfortunately, due to a synchronization bug with Google sync, I have lost the historic list of accuracies and losses, and I don'y have enough time to rerun the model.
            - If I could describe the learning curve myself, it was like (for every 10K examples): 40%, 46%, 53%, 57% than after 1 whole epoch roughly 69%, 7 more epochs got it to 74.8%, and than any more training kept it in 74.8%
- What was involved in replcating the result?
    - If you tried several hyperparameters, or several approaches, describe them, and what was the result of each.
        - Well at first, I read the article and understood that the model is: `Embedding -> F -> Score -> G -> Sum -> H -> Final`
        and so my first implementation was quite simple, F, G, H, Final where 1 matrix each, and the embedding was not projected.
        Then, I understood that the embedding needed projection, as described in the article (use glove, 300x200), but out of no real reason I chose to keep all of my layers with a 300x300 matrix, probably just because I wanted simplicity.
        I did not manage to get anywhere above 61%, so I read the pytorch implementation, and there I saw they made the F, G, and H layers more complex, by making each of them like: `relu(W1*dropout(relu(W2*dropout(X))))` so I implemented that as well.
        Along with a change to my final layer (I accidentally called softmax on the final layer twice) I got to this model, but it didn't converge fast enough.
        I noticed that using word2vec (dependencies) gave better coverage for the data vocabulary than glove by a lot, so I switched.
    - If you didn't manage to replicate the result, describe your attempts in details.
        - Other than the above, I didn't understand why my results are not the same as theirs. I even went as far as translating their PyTorch code exactly to DyNet, but no luck there.
        I believe, that they are doing something nasty here. If you look at their code, first they use the preprocessed files that clean the data, and remove a lot of sentences. Then, they do something cheaty, and remove all sentences containing more than 10 words.
        I also really didn't like that they were using Adagrad with custom weight decay, and stowed away of that as I don't know what is the incentive there.
    - I did not attempt the preprocessing, as it feels like cheating.
    - I contacted the creator of the PyTorch implementation, and they really strongly pointed out that I should change my "weight clipping, l2
regularisation", but I did not feel comfortable enough with my understanding of how things are done there, and with let's say at least 4 hours until I can see if my model is doing alright, I just couldn't change stuff as I please.
- What worked straightforward out of the box? what didn't work?
    - The data was super nice to work with
    - The model never errored on me, the multiplications were clear from the paper so it was straight forward to get the graph done.
    - Glove was not a good choice, instead, dependencies word embedding worked a lot better with better vocab coverage and faster convergence.
- Are there any improvements to the algorithm you can think about?
    - Since the "Score" part, there are so many non linear steps, that I am afraid the algorithm is leaking some data it found.
    I would suggest moving some of that data in a linear way to after the G step, and then doing the same with the "Sum" part.