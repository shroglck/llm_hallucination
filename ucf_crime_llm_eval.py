import openai
import time
import os
import argparse
import json
import ast
import json
from multiprocessing.pool import Pool

openai.api_key = "358f737e5bcf4b79a064fda9fe97032b"
import os
from openai import AzureOpenAI
client = AzureOpenAI(
api_key = "358f737e5bcf4b79a064fda9fe97032b",  
api_version = "2024-02-01",
azure_endpoint = "https://instanceucf.openai.azure.com/"
)

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    args = parser.parse_args()
    return args

def main(question, answer,pred,goal):
    #question = "If the intended goal of the action taking place in the provided video is given as : Two people were driving down a path with a camera on a stick..Describe why the the action fails?"
    #answer = "They lifted the camera too high and hit a branch falling off."
    #pred = "The video shows two people driving down a path with a camera on a stick. The main goal of the action is to capture the scenery and the path they are driving on. However, the video deviates from the original objective as it shows the camera being knocked over and the people driving off the path. The video ends with the camera being thrown into the air. The most probable reason for the failure of this action is that the person holding the camera lost control of it, causing it to fall and get knocked over. This could be due to a sudden change in the path, an unexpected obstacle, or the person's inexperience in handling the camera."
    
    if True:
        completion = client.chat.completions.create(
                    model="gpt35",
                    messages=[
                        {
                            "role": "system",
                            "content": 
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs with the ground truth answer. "
                                "You are provided the ground truth description of the crime occuring in the video, the main crime and the predicted answer. The correct answer describes the event which describes the crime occuring in the video. Your task is to evalute the correctness of the predicted answer. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Make sure that the crime taking place in the predicted answer is same as the main crime occuring in the video.\n "
                                
                                "- Focus on the meaningful alignment of the events and activities between the predicted answer and the ground truth description.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Main Crime: {question}\n"
                                f"Ground truth description : {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
        response_message = response.model_dump_json(indent=2)
        response_dict = ast.literal_eval(response_message)
    else:
        d=1
     #   resposnse_dict = {}
      #  response_dict['pred'] = 
    return response_dict
    
    
    

if __name__ == "__main__":
    
    args = parse_args()
    
    f = json.load(open(args.pred_path))
    with open('/home/sh009885/code/oops_dataset/annotations/transition_times.json') as json_file:
        gt_contents = json.load(json_file)
    svo_path  = "/home/sh009885/code/oops_dataset/annotations/svos/val.json"
    with open(svo_path) as l:
        svo_data = json.load(l)
        svo_data = {k.encode("cp1252").decode("utf-8"): v for k, v in svo_data.items()}
    answers = []
    cnt = 0
    vide = []
    if False:
        with open(os.path.join(args.output_dir, f"{args.output_json}.json")) as saved_file:
            opened_file = json.load(saved_file)
        for n in opened_file:
            vide.append(n["video_name"])
        answers = opened_file
        print(len(opened_file))
    else: 
        b=1
    #
    with open("ckpts/ucf_crime_ann.json") as saved:
        ans_key = json.load(saved)
    ans_key_l = []
    video_names = []
    for an in ans_key:
        vid_name = an["video_name"]
        an["video_name"] = "/home/c3-0/datasets/UCF_Crimes/Videos/"+vid_name
        video_names.append(an["video_name"])
        ans_key_l.append(an)
    for t,i in enumerate(f):
        cnt+=1
        print(len(answers))
        #if cnt<50:
        #    continue
        if True:
            if "video_name" not in i.keys():
                continue
            video_name = i["video_name"]
            tor = None
            for tir in ans_key_l:
                if video_name in tir["video_name"]:
                    tor = tir
                    #print(video_name,tor)
                    break
            #if tor is None:
            #    continue
            #print(tor)
            
            #l = svo_data[video_name[:-4]][0]
            ans = tor["pred"][0]
            question = tor["video_name"].split("/")[6]#f"Describe the criminal activity that occurs in the video?"
            ans_dict = {}
            ans_dict["video_name"] = video_name
            ans_dict["video_name_target"]= tor["video_name"]
            ans_dict["pred"] = []
            ans_dict["score"] = []   
            #if video_name in vide :
             #   continue
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

            
        else:
            b=1
        
        #ans_dict = main()
