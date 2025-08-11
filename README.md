# DomainCoT:

This is proposed to enhance LLM reasoning with rules provided by domain-experts.

## Model Choice:

1) Since we want to manipulate the thinking process, we do not choose those models which are specifically trained for outputing CoT and result.
Instead, we would like to implement Tree-of-Thought manually. (Already implemented by Xuanqi)

2) To utilize the reasoning LLM (e.g., Deepseek R1), we need to know what is the start/end in the reasoning step, and how to have multiple candidates next steps given the curret prompt/cot prefix.

## Dataset Choice:
1) BBEH(https://github.com/google-deepmind/bbeh)
to-be-continued...


