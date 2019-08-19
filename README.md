# GraphSW

This repository is the implementation of GraphSW ([arXiv](https://arxiv.org/abs/1908.05611)):

![image](https://github.com/mengruwu/graphsw/blob/master/framwork.png)
![image](https://github.com/mengruwu/graphsw/blob/master/performance.png)

# GraphSW

This repository is the implementation of GraphSW ([arXiv](https://arxiv.org/abs/1908.05611)):

![image](https://github.com/mengruwu/graphsw/blob/master/framwork.png)



### Files in the folder

- `data/`: datasets
  - `Book-Crossing/`
  - `MovieLens-1M/`
  - `amazon-book_20core/`
  - `last-fm_50core/`
  - `music/`
  - `yelp2018_20core/`
- `src/`: implementation of GraphSW.
- `output/`: storing log files
- `misc/`: storing users being evaluating, popular items, and sharing embedding.

### Build the environment(conda)

```
$ cd graph-stage
$ conda deactivate
$ conda create -f requirements.yml
$ conda activate graph-stage
```
- RippleNet
  ```
  $ cd src/RippleNet/
  $ bash RippleNet_{dataset}.sh
  ```
- KGCN
  ```
  $ cd src/KGCN/
  $ bash main_{dataset}.sh
  ```
