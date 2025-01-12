# 2025-01-11 

Forked repo

- make sure I'm testing on test
- [x] run
- [ ] replicate
- [ ] compare the 3 implementations
- [ ] experiments
  - [ ] always think
  - [ ] use supressed activations used for hidden state only
  - [ ] make the latent space sparse, interpretable, compressed etc?


Wow 25GB or gpu ram was not enougth?

This page says, 17.22 GB of GPU RAM. to fine tune a 1b model https://lightning.ai/lightning-ai/studios/finetune-and-serve-llama-3-2-1b-and-3b  presumably 2b is ~40. I know you can train in 8bit though, and use adam 8b

But I eventually found it takes 25gb, but that's with a 128 seq len



- [ ] read full nb
- [ ] compare 3 impl
- [ ] try my ideas


# 2025-01-12

Hmm I'm not sure this repo is setup in the way I'd like
- The synthetic CoT doesn't really make sense
- The current results are on the train set
- Translation is not the best task to show this on, math is better
- altho it's nice to use gemeni, and to see everything set out in a notebook, and to have a nicely commented class
