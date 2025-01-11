# 2025-01-11 

Forked repo

- make sure I'm testing on test
- [ ] replicate
- [ ] compare the 3 implementations
- [ ] experiments
  - [ ] always think
  - [ ] use supressed activations used for hidden state only
  - [ ] make the latent space sparse, interpretable, compressed etc?


Wow 25GB or gpu ram was not enougth?

This page says, 17.22 GB of GPU RAM. to fine tune a 1b model https://lightning.ai/lightning-ai/studios/finetune-and-serve-llama-3-2-1b-and-3b  presumably 2b is ~40. I know you can train in 8bit though, and use adam 8b

But I eventually found it takes 25gb, but that's with a 128 seq len
