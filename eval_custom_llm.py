import openai
import time
import os
import argparse
import json
import ast
import json
from multiprocessing.pool import Pool

openai.api_key =""#
def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    args = parser.parse_args()
    return args

def main(question, answer,pred,goal):
    
    try:
        response = client.chat.completions.create(
                    model="gpt35",
                    messages=[
                        {
                            "role": "system",
                            "content": 
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "You are provided with a question,the correct answer and the predicted answer. The question contains information about the task being attempted to be achieved in the video, along with the context about the objects involved in achieving that goal. The correct answer consists of the reasons behind the failure of achieving that objective and information about the objects present during the failure. Your task is to evalute the correctness of the predicted answer. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                
                                "- Focus on the meaningful match of events between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness and alignment of the predicted answer compared to the correct answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
        )
                # Convert response to a Python dictionary.
        #response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        
        #response_message = completion["choices"][0]["message"]["content"]
        #response_dict = ast.literal_eval(response_message)
        response_dict = response.model_dump_json(indent=2)
        #print(response_dict)
        #print(response.choices[0].message.content)
        #print(response_dict)
    except:
        d=1
     #   resposnse_dict = {}
      #  response_dict['pred'] = 
    return ast.literal_eval(response.choices[0].message.content)
    
    
    

if __name__ == "__main__":
    
    args = parse_args()
    
    f = json.load(open(args.pred_path))
    with open('/home/sh009885/code/oops_dataset/annotations/transition_times.json') as json_file:
        gt_contents = json.load(json_file)
    svo_path  = "/home/sh009885/code/oops_dataset/annotations/svos/val.json"
    with open(svo_path) as l:
        svo_data = json.load(l)
        svo_data = {k.encode("cp1252").decode("utf-8"): v for k, v in svo_data.items()}
    json_data = json.load(open("/home/sh009885/code/videos/dataset.json"))
    answers = []
    cnt = 0
    vide = []
    try:
        with open(os.path.join(args.output_dir, f"{args.output_json}.json")) as saved_file:
            opened_file = json.load(saved_file)
        for n in opened_file:
            vide.append(n["video_name"])
        answers = opened_file
        print(len(opened_file))
    except: 
        b=1
    #
        
    for t,i in enumerate(f):
        cnt+=1
        #print(len(answers))
        try:
            video_name = i["video_name"]
            #l = svo_data[video_name[:-4]][0]
            #goal = l["goal"]
            for l in json_data:
                if video_name == l["video_name"]:
                    ans = l["wrong"]
            ans = i["actual_wrong"]
            question = f"Describe why the the action fails?"
            ans_dict = {}
            ans_dict["video_name"] = video_name
            ans_dict["pred"] = []
            ans_dict["score"] = [] 
            #i["pred"] = i["response"]
            #if video_name in vide:
            #    continue
            vide.append(video_name)
            for j in i["pred"]:
                tst = j.split(' ')
                if len(tst)<100:
                    ans_d= main(question,ans,j,"")
                else:
                    ans_d = {}
                    ans_d["pred"] = "no"
                    ans_d["score"] = 0 
                ans_dict["pred"].append(ans_d["pred"])
                ans_dict["score"].append(ans_d["score"])
                #print(ans_dict)
                time.sleep(1)
            answers.append(ans_dict)
            if (t+1)%1 == 0:
                with open(os.path.join(args.output_dir, f"{args.output_json}.json"), 'w') as file:
                    json.dump(answers, file)
                    #answers = []
                    time.sleep(4)

            
        except:
            b=1
        
        #ans_dict = main()
