<div align="center">

<p align="center">
  <img src="asset/logo.png" alt="Logo" width="25%">
</p>

<h1>LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs</h1>

![AVQA](https://img.shields.io/badge/Task-AudioVideo--QA-red) 
![Long-Video](https://img.shields.io/badge/Task-LongVideo--Understanding-red) 
![LVOmniBench](https://img.shields.io/badge/Dataset-LVOmniBench-blue)
![Gemini](https://img.shields.io/badge/Model-Gemini-green) 

<font size=7><div align='center' > [[Project Page](https://kd-tao.github.io/LVOmniBench/)] [[Paper](https://arxiv.org/abs/2603.19217)] [[Dataset](https://huggingface.co/datasets/KD-TAO/LVOmniBench)]  </div></font>

LVOmniBench is a new audio-visual understanding evaluation benchmark in long-form audio-video inputs. 🌟

</div>



## 🔥 News
* **`2026.03.19`** 🌟 We are very proud to launch LVOmniBench, the pioneering comprehensive evaluation benchmark of OmniLLMs in Long Audio-Video Understanding Evaluation!



## ✨ LVOmniBench Introduction

Recent advancements in omnimodal large language models (OmniLLMs) have significantly improved the comprehension of audio and video inputs. However, current evaluations primarily focus on short audio and video clips ranging from 10 seconds to 5 minutes, failing to reflect the demands of real-world applications, where videos typically run for tens of minutes. To address this critical gap, we introduce LVOmniBench, a new benchmark designed specifically for the cross-modal comprehension of long-form audio and video.


* We curated a diverse collection of long videos, with durations ranging from
**10 to 90 minutes** and an average duration of **2,069s**. This duration represents
a greater than sixfold increase in temporal scale compared to that of existing
benchmarks for audio-visual understanding.

*  We **manually constructed 1,014
high-quality multiple-choice questions**, which are explicitly designed to require
joint reasoning across the audio and visual modalities, thereby facilitating a more
comprehensive evaluation of OmniLLMs.

*  Each QA is ranked by difficulty level, and long audio-video understanding poses significant challenges for both current proprietary and open source models!


<p align="center">
    <img src="asset/question_bar_00.png" width="100%" height="100%">
</p>

## 🌰 Dataset Examples

<p align="center">
    <img src="asset/teaser_case_00.png" width="100%" height="100%">
</p>


## 🔮 Evaluation

📍 **Prompt**:

The common prompt used in our evaluation follows this format:

```python
prompt_text = (
    f"Question: {question}\n"
    f"Options:\n{options_str}\n\n"
    "Select the best answer from the options above. "
    "Directly provide the letter representing your choice (A/B/C/D) and nothing else. "
    "Do not include the full text of the option, do not provide any explanation."
)
```

📍 **Leaderboard**: 

If you want to add your results to our LVOmniBench leaderboard, please contact us at **taokeda@westlake.edu.cn**


## 🏆 Experimental Results
- **Evaluation results of different OmniLLMs.**

<p align="center">
    <img src="asset/results_main.png" width="90%" height="50%">
</p>


- **Evaluation results across different task types.**

<p align="center">
    <img src="asset/results_type.png" width="90%" height="50%">
</p>


<!-- ## Related Works

#### [Awesome Audio-Visual Understanding Benchmark] Here, we summarize the existing audio-visual understanding benchmarks for OmniLLMs. -->




## 🌍 Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
coming soon
```

