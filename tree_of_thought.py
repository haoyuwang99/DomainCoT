import re
import heapq 
import transformers
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.output_parsers import PydanticOutputParser
import numpy as np
import pandas as pd
import uuid
import json
from langchain_util import *
from pydantic import BaseModel, Field
from enum import Enum

class ReasoningStep(BaseModel):
    step: str = Field(description="Next logical step or intermediate result to advance the solution.")
    finished: bool = Field(description="Whether the chain-of-thought is finished after the generated step")

class ReasoningStepCandidates(BaseModel):
    pass

def generate_next_reasoning_step(input_text, previous_steps, llm):
    step_text = "N/A"
    if len(previous_steps) > 0: 
        step_text = ""
        for idx, step in enumerate(previous_steps, start=1):
            step_text += f"{idx}. {step}\n"
            
    prompt = f"""You are an expert reasoner who thinks in structured trees of possibilities before deciding.
Solve the problem: {input_text}
Previous steps: {step_text}
State the next logical step or intermediate result to advance the solution. Indicate if the reasoning chain ends here. 
"""
    llm_with_output_parser = llm.bind_tools([ReasoningStep])
    output = llm_with_output_parser.invoke(prompt)
    r_step = ReasoningStep.model_validate(output.tool_calls[0]["args"])
    return r_step.step, r_step.finished


def generate_answer(llm, input, reasoning_steps, output_format, formatter):
    step_text = "N/A"
    for idx, step in enumerate(reasoning_steps, start=1):
        step_text += f"{idx}. {step}\n"
    prompt = f"""You are an expert at problem solving.
Given the problem: {input}
Following the reasoning steps:\n {reasoning_steps}
What result of the problem?
{output_format}
"""
    llm_with_output_parser = llm.bind_tools([formatter])
    output = llm_with_output_parser.invoke(prompt)
    result = formatter.model_validate(output.tool_calls[0]["args"])
    return result.solution
 
def print_tree_structure(tree, title="Tree Structure"):
    print(f"\n--- {title} ---")
    if not tree.nodes:
        print("Tree is empty.")
        return

    # Simple BFS traversal for printing
    queue = [(tree.root_id, 0)]
    visited = set()

    while queue:
        node_id, depth = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node = tree.get_node(node_id)
        if node:
            indent = "  " * depth
            score_info = f" (Score: {node.score:.2f})  Consistency: {node.consistency:.2f} ) \t " if node.score is not None else ""
            print(f"{indent}- [{node.id[:8]}] {node.text.split('.')[0]}{score_info}")
            for child_id in node.children_ids:
                queue.append((child_id, depth + 1))
        else:
            print(f"{'  ' * depth}- [Node {node_id[:8]} not found]")

# this version iteratively generate chain-of-thought via LLM and solve the problem. 
def solve_with_raw_cot(llm, input_text,  output_format, formatter, max_depth=5):
    finish = False
    reasoning_steps = [] 
    depth = 0
    while depth <= max_depth and not finish:
        step, finish = generate_next_reasoning_step(input_text, reasoning_steps, llm)
        reasoning_steps.append(step)
        depth = depth + 1
    answer = generate_answer(llm, input_text, reasoning_steps, output_format, formatter)
    return answer

EXAMPLE_INPUT = """
A few players are playing a boardgame. The current state of the game is as follows. 
The bee has a football with a radius of 15 inches. The chihuahua has a banana-strawberry smoothie. 
The chihuahua lost her keys. 
The cobra suspects the truthfulness of the fangtooth. 
The cougar is named Tarzan. 
The crab has two friends that are wise and 1 friend that is not. 
The crab is named Tessa. The crow tears down the castle that belongs to the chihuahua. 
The dalmatian borrows one of the weapons of the fish. The duck is a marketing manager. 
The seal pays money to the bulldog. The shark brings an oil tank for the dove. 
The stork has a card that is white in color, and is watching a movie from 1924. 
The vampire tears down the castle that belongs to the ostrich. 
The wolf has a card that is indigo in color. 
The zebra borrows one of the weapons of the mouse. 
The butterfly does not tear down the castle that belongs to the dragon. 
The fish does not swim in the pool next to the house of the pelikan. 
The gadwall does not unite with the monkey. The goose does not borrow one of the weapons of the chihuahua. 
And the rules of the game are as follows.
Rule1: The stork will not pay money to the beaver if it (the stork) is watching a movie that was released after world war 2 started. 
Rule2: If the beetle does not suspect the truthfulness of the finch, then the finch manages to convince the ant. 
Rule3: One of the rules of the game is that if the dachshund shouts at the leopard, then the leopard will never shout at the finch. 
Rule4: If you are positive that one of the animals does not leave the houses that are occupied by the husky, you can be certain that it will not create a castle for the badger. 
Rule5: If the stork has a basketball that fits in a 23.6 x 22.1 x 26.8 inches box, then the stork pays some $$$ to the beaver. 
Rule6: The stork will not pay money to the beaver if it (the stork) has a card whose color appears in the flag of Japan. 
Rule7: If at least one animal dances with the camel, then the songbird shouts at the reindeer. 
Rule8: The crab will not dance with the camel if it (the crab) has a name whose first letter is the same as the first letter of the cougar's name. 
Rule9: The wolf will swim inside the pool located besides the house of the goat if it (the wolf) has a notebook that fits in a 21.7 x 17.4 inches box. 
Rule10: If something does not swim in the pool next to the house of the pelikan, then it smiles at the flamingo. 
Rule11: The songbird does not shout at the reindeer, in the case where the swan neglects the songbird. 
Rule12: The rhino unquestionably pays money to the finch, in the case where the fish shouts at the rhino. 
Rule13: One of the rules of the game is that if the butterfly does not tear down the castle of the dragon, then the dragon will, without hesitation, tear down the castle of the worm. 
Rule14: From observing that an animal does not swim inside the pool located besides the house of the goat, one can conclude that it neglects the reindeer. 
Rule15: If you are positive that you saw one of the animals suspects the truthfulness of the seal, you can be certain that it will also fall on a square that belongs to the poodle. 
Rule16: For the badger, if you have two pieces of evidence 1) the stork brings an oil tank for the badger and 2) the chihuahua does not create one castle for the badger, then you can add that the badger will never take over the emperor of the german shepherd to your conclusions. 
Rule17: The living creature that does not pay some $$$ to the beaver will never bring an oil tank for the badger. 
Rule18: If the badger shouts at the finch, then the finch is not going to manage to persuade the ant. 
Rule19: This is a basic rule: if the dalmatian borrows one of the weapons of the fish, then the conclusion that "the fish will not smile at the flamingo" follows immediately and effectively. 
Rule20: If something leaves the houses that are occupied by the mule, then it dances with the dragonfly, too. 
Rule21: This is a basic rule: if the seal pays money to the bulldog, then the conclusion that "the bulldog suspects the truthfulness of the seal" follows immediately and effectively. 
Rule22: If you are positive that one of the animals does not take over the emperor of the german shepherd, you can be certain that it will shout at the finch without a doubt. 
Rule23: There exists an animal which unites with the akita? Then the fish definitely shouts at the rhino. 
Rule24: This is a basic rule: if the woodpecker destroys the wall constructed by the dachshund, then the conclusion that "the dachshund shouts at the leopard" follows immediately and effectively. 
Rule25: The wolf will swim inside the pool located besides the house of the goat if it (the wolf) has a card whose color starts with the letter "n". 
Rule26: If something does not pay money to the swallow, then it unites with the akita. 
Rule27: There exists an animal which falls on a square that belongs to the poodle? Then the frog definitely dances with the bear. 
Rule28: Here is an important piece of information about the chihuahua: if it has a card whose color is one of the rainbow colors then it leaves the houses occupied by the husky for sure. 
Rule29: Here is an important piece of information about the chihuahua: if it has a leafy green vegetable then it leaves the houses occupied by the husky for sure. 
Rule30: If at least one animal negotiates a deal with the liger, then the stork brings an oil tank for the badger. 
Rule31: The elk does not build a power plant near the green fields of the rhino whenever at least one animal calls the walrus. 
Rule32: If something creates one castle for the bear, then it does not dance with the bear. 
Rule33: There exists an animal which suspects the truthfulness of the fangtooth? Then the crab definitely dances with the camel. 
Rule34: If something does not manage to convince the ant but stops the victory of the beetle, then it will not shout at the mermaid. 
Rule35: The duck will not call the walrus if it (the duck) works in computer science and engineering. 
Rule36: If you are positive that you saw one of the animals shouts at the dolphin, you can be certain that it will not pay money to the finch. 
Rule37: If the chihuahua does not have her keys, then the chihuahua does not leave the houses occupied by the llama. 
Rule38: Here is an important piece of information about the bee: if it has a football that fits in a 40.3 x 34.3 x 34.3 inches box then it negotiates a deal with the liger for sure. 
Rule39: If the wolf neglects the reindeer, then the reindeer trades one of its pieces with the leopard. 
Rule40: There exists an animal which dances with the bear? Then, the beetle definitely does not suspect the truthfulness of the finch. 
Rule41: The gadwall leaves the houses that are occupied by the mule whenever at least one animal borrows a weapon from the mouse. 
Rule42: If there is evidence that one animal, no matter which one, tears down the castle of the ostrich, then the wolf is not going to swim in the pool next to the house of the goat. 
Rule43: If at least one animal dances with the dragonfly, then the rhino reveals a secret to the otter. 
Rule44: If the fish smiles at the flamingo, then the flamingo is not going to pay money to the swallow. 
Rule45: This is a basic rule: if the crow tears down the castle that belongs to the chihuahua, then the conclusion that "the chihuahua will not leave the houses occupied by the husky" follows immediately and effectively. 
Rule46: There exists an animal which brings an oil tank for the dove? Then the duck definitely calls the walrus. 
Rule47: Here is an important piece of information about the duck: if it has more than three friends then it does not call the walrus for sure. 
Rule48: The woodpecker destroys the wall built by the dachshund whenever at least one animal tears down the castle that belongs to the worm. 
Rule49: One of the rules of the game is that if the leopard does not shout at the finch, then the finch will, without hesitation, stop the victory of the beetle.
Rule10 is preferred over Rule19. 
Rule11 is preferred over Rule7. 
Rule18 is preferred over Rule2. 
Rule25 is preferred over Rule42. 
Rule28 is preferred over Rule45. 
Rule29 is preferred over Rule45. 
Rule30 is preferred over Rule17. 
Rule32 is preferred over Rule27. 
Rule33 is preferred over Rule8. 
Rule35 is preferred over Rule46. 
Rule36 is preferred over Rule12. 
Rule47 is preferred over Rule46. 
Rule5 is preferred over Rule1. 
Rule5 is preferred over Rule6. 
Rule9 is preferred over Rule42. 
A rule is only applicable if all of its antecedents can be proved. If a rule is preferred over the other, it means whenever both of them can be applied to derive new conclusions and those conclusions contradict with each other (e.g., from one we derive X and from the other we derive not X), we should go with the conclusion from the rule with higher preference. 
Based on the facts, rules, and preferences, what is the truth value of the statement, does the finch shout at the mermaid? 
"""
OUPUT_FORMAT = "Answer 'proved' if it can be proved, 'disproved' if it can be disproved, and 'unknown' if it can neither be proved nor disproved."
ANSWER = "disaproved"

class Solution(str, Enum):
    PROVED = "proved"
    UNPROVED = "unproved"
    UNKNOWN = "unknown"

class Result(BaseModel):
    solution : Solution

import time
if __name__ == "__main__":
    
    llm = get_openai_llm()
    answer = generate_answer(llm, EXAMPLE_INPUT,[], OUPUT_FORMAT, Result)
    print(answer.value)
    # answer = solve_with_cot_reason(llm, EXAMPLE_INPUT, OUPUT_FORMAT, Solution)
    
