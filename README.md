# build_nanogpt
following along https://www.youtube.com/watch?v=l8pRSuU81PU


# misc notes from video
* when trying to overfit on tinyshakespeare, we expect to see big loss improvement from first few batches because most of the 50k tokens aren't being used, and it can immediately drive most of those to 0
* GPT3 gradually increases batch size linearly initially. intuition is that initially, you're just learning what tokens are used frequently, and for that basic task, different batches are extremely correlated. no point of larger batches initially
* data sampled without replacement per epoch