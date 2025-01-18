import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import argparse
import random
import json
import os
from tqdm import tqdm


model_path = 'LanguageBind/Video-LLaVA-7B'
device = 'cuda'
    
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
def video_list(base):
    cls = os.listdir(base)
    indi_vid = [os.listdir(os.path.join(base,i)) for i in cls]
    all_videos =[]
    for o,i in enumerate(indi_vid):
        for j in i:
            all_videos.append(os.path.join(base,cls[o],j))
    return all_videos

def main(path,prompt):
    disable_torch_init()
    video = path
    inp = prompt
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    key = ['video']

    #print(f"{roles[1]}: {inp}")
    inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[tensor, key],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=2048,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs
    
    
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--num_options", type = int,default = 4)
    parser.add_argument("--goal",action="store_true" )
    parser.add_argument("--FIB",action="store_true"  )
    parser.add_argument("--REASON",action="store_true" )
    parser.add_argument("--res",action="store_true" )
    parser.add_argument("--rand",action="store_true")
    parser.add_argument("--y_n",action="store_true" )
    parser.add_argument("--zero",action="store_true")
    parser.add_argument("--no_combine",action="store_true" )
    parser.add_argument("--int_sparse",action="store_true"  )
    parser.add_argument("--unint_sparse",action="store_true" )
    parser.add_argument("--rand_vid",action="store_true" )
    parser.add_argument("--num_frms",type = int , default = 100)
    
    return parser.parse_args()
    
    
    
    
    
def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    # Load the ground truth file
    #with open(args.gt_file) as file:
    #    gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    with open('/home/sh009885/code/oops_dataset/annotations/transition_times.json') as json_file:
        gt_contents = json.load(json_file)
    svo_path  = "/home/sh009885/code/oops_dataset/annotations/svos/val.json"
    with open(svo_path) as f:
        svo_data = json.load(f)
        svo_data = {k.encode("cp1252").decode("utf-8"): v for k, v in svo_data.items()}
    # Iterate over each sample in the ground truth file
    num_options = args.num_options
    cnt =0 
#    all_videos = video_list("/home/c3-0/datasets/UCF_Crimes/Videos")

    an = video_list("/home/c3-0/datasets/UCF_Crimes/Videos")
    an = json.load(open("/home/sh009885/code/Video-ChatGPT/ckpts/ucf_crime_ann.json"))
    an = ["/home/c3-0/datasets/UCF_Crimes/Videos/"+i["video_name"]for i in an ]
    
    for sample_dict in tqdm(svo_data):
        cnt+=1
        if cnt> len(an):
            break
        
        
        ans_list =[]
        dummy1 = random.choice(list(svo_data.keys()))
        #for i in range(num_options):
        #    dummy1 = random.choice(list(svo_data.keys()))
        #   w_opt1 = svo_data[dummy1][0]['wentwrong']
        #    ans_list.append(w_opt1)
        sample = sample_dict#list(sample_dict.keys())[0]
        data = svo_data[sample][0] 
        went_wrong = data['wentwrong']
        actual_wrong = data['wentwrong']
        svos=[]
        for i in data['kwentwrongsvos']:
            if i not in svos and len(i)>1:
                svos.extend(i[1:])
        witout = []
        for i in svos:
            if i not in witout and "The" not in witout or "the" not in witout:
                witout.append(i)
        if len(witout)<=1:
            continue
            
        no_to_replace = random.randint(1,len(witout)-1)
        replace_str=""
        ans  = []
        for i in range(1,len(witout)):
            no_ = random.randint(1,len(witout)-1)
            replaced = witout[i]
            for j in went_wrong.split(' '):
                if replaced in j:
                    replace_str="_"*len(j)
                    ans.append(j)
                    break
            went_wrong = went_wrong.replace(j,replace_str)
        
        #dummy_video = svo_data[dummy]
        #dummy_video = dummy+".mp4"
        video_name = sample+".mp4"
        #if args.rand_vid:
        #    video_name = dummy1+".mp4"
        sample_set = {}
        #question = sample['Q']
        #question = "Is the action performed in the video intentional"
        split_ratio  = gt_contents[sample]['rel_t'][0]
        split_ratio_ = gt_contents[sample]['rel_t'][0]
        d = svo_data[sample]
        goal = d[0]['goal']
        actual_goal = goal
        actual_fail = d[0]['wentwrong']
        #ans_list = [actual_fail,w_opt1,w_opt2,w_opt3]
        ans_list.append(actual_fail)
        ans_list.append("The action is executed successfully")
        random.shuffle(ans_list)
        options_list = ""
        ans_list = ["Yes","No"]
        random.shuffle(ans_list)
        if ans_list[0]=="Yes":
            yes_ans = 0
            no_ans = 1
        else:
            yes_ans =1
            no_ans = 0
        for i,j in enumerate(ans_list):
            #if i ==  actual_fail:
                #correct_ans  = str(i)
            options_list += str(i)+":"+" "+j+".  "
        #print(" +++++++++++++++++++++++++++++++++")
        #print(options_list)
        #print(" +++++++++++++++++++++++++++++++++")
        #A,B,C,D = ans_list[0],ans_list[1],ans_list[2],ans_list[3]
        #if A == actual_fail:
        #    correct_ans = 'A'
        #elif B == actual_fail:
        #    correct_ans = 'B'
        #elif C == actual_fail:
        #    correct_ans = 'C'
        #elif D == actual_fail:
        #    correct_ans = 'D'

        #args.goal = False
        #question = f"In the given video the an action is being performed with the goal of achieving {goal}. Can you describe accurately what is the reason why the action is unintentional by describing its deviation from normal goal?. Also does the deviation of the action from the original goal make the action funny?. Focus on why the actions might fail like objects involved or actor commiting an error and why that error occurs.Focus on how the action could be funny given that it fails to achieve its goal.Provide most probable response."
        #question =f"In the given video the an action is being performed with the goal of achieving {goal}. Can you describe accurately what is the reason why the action is unintentional by describing its deviation from normal goal? You will be given four options A,B,C,D and you have to select the correct option which describes the event that led to the deviation from the goal in best possible manner. The four options are A-{A}\n B-{B}\n C-{C} \n D-{D}\n" 
        if args.y_n and args.goal:
            question = f"If the intended goal of the action taking place in the provided video is given as : {goal}.From the options select the option which describes most accurately whether the action is completed as intended or not.Options : {options_list}. Note: Answer should not exceed 10 words.Answer consists of the correct option only i.e. Yes or No with no other explanation needed"
        elif args.y_n:
            question = f"From the options,using the information present in the video, select the option which describes most accurately whether the action is completed as intended or not. Options : {options_list}. Note: Answer should not exceed 10 words.Answer consists of the correct option only i.e. Yes or No with no other explanation needed"
        elif args.FIB and args.goal:
            question = f"The intended goal of the action taking place in the provided video is given as : {goal} Given the following video and the information about the goal of the action being performed perform the following task. Complete the following sentence such that the sentece describes the reasoning behind failure of the intended action in the video. The sentence to be completed is {went_wrong} Note: Your task is to complete the given sentence where the blanks are indicated by _____."
            split_ratio = -1
        elif args.FIB and args.goal == False:
            question = f"Given the following video complete the following sentence such that the sentece describes the reasoning behind failure of the intended action in the video. The sentence to be completed is {went_wrong} Note: Your task is to complete the given sentence where the blanks are indicated by _____."
            split_ratio = -1

            
        
        elif args.REASON and args.goal:
            question = f"the intended goal of the action taking place in the provided video is given as : {goal}.Describe why the action fails to achieve the intended goal."
            split_ratio = -1
        elif args.res :
            question = f"Describe why the action action taking place in the video fails. NOTE: The output has to be limited to 10 words. "
            split_ratio=1
        #print(question)
            
            
        #question = f"In the given video the an action is being performed.Can you describe accurately what is the reason why the action is unintentional by describing its deviation from normal goal?. Also does the deviation of the action from the original goal make the action funny?. Focus on why the actions might fail like objects involved or actor commiting an error and why that error occurs.Focus on how the action could be funny given that it fails to achieve its goal.Provide most probable response."
        #question =f"Does the action taking place in the given video fail? You will be given {num_options}  describing the reasoning behind the failure. The options for this video are given as {options_list} . The options are given in for a possible reason and the numebr of associated with it like '1: Reason behind failure'. Provide only the answer number related with the correct answer for example if 3rd option is correct the answer should just be 3."
        
 
        #question = f"given the following video complete the following sentence such that the sentece describes the video. The sentence to be completed is {went_wrong}'"
        #print(question,ans)
        #question
        #split_ratio = -1
        if split_ratio <=0:
            split_ratio = None
        # Load the video file
        for fmt in video_formats:  # Added this line
            args.video_dir = "/home/sh009885/code/oops_dataset/oops_video/val"
            temp_path = os.path.join(args.video_dir, f"{video_name}")
            if os.path.exists(temp_path):
                
                video_path = temp_path
                break
            else:
                args.video_dir = "/home/sh009885/code/oops_dataset/oops_video/train"
                video_path = os.path.join(args.video_dir, f"{video_name}")
                break
                

        # Check if the video exists
        args.video_dir = "/home/sh009885/code/oops_dataset/oops_video/val"
                
        video_path = os.path.join(args.video_dir,video_name)
        
        #question = "Given the video explain why the intended action ends up in failure."
        #print(video_path)
        correct_ans = actual_fail
        video_frames_1,video_frames_2 = None,None
        ans1,ans2=None,None
        
        split_ratio = None
        #print(video_path)
        #print(args)
        
        
        
        if False :
            if (video_path is not None) and (split_ratio is not None):  # Modified this line
                video_frames_1 = load_video(video_path,end = split_ratio,rand=args.rand,zero =args.zero,num_frm = args.num_frms)
                ans1 = [correct_ans,"Yes"]

                    #ans1 = ans
                video_frames_2 = load_video(video_path,start = split_ratio,rand=args.rand,zero = args.zero,num_frm = args.num_frms)
                        #ans2 = correct_ans
                ans2 = [correct_ans,"No"]
                #prin("one")

            elif video_path is not None:
                if args.no_combine:
                    video_frames_1 = load_video(video_path,start = split_ratio,rand=args.rand,zero =False,num_frm = args.num_frms)
                            #ans1 = correct_ans
                    ans1 = [correct_ans,"Yes"]
                    #print("two")
                else:
                    if args.int_sparse and not(args.unint_sparse):
                        video_frames_1 = load_video(video_path,end = split_ratio_,rand=args.rand,zero =args.zero,num_frm=10)
                        ans1 = [correct_ans,"Yes"]
                        video_frames_2 = load_video(video_path,start = split_ratio_,rand=args.rand,zero = args.zero,num_frm = 90)
                        #ans2 = correct_ans
                        video_frames_1.extend(video_frames_2)
                    
                    elif not(args.int_sparse) and not(args.unint_sparse):
                        video_frames_1 = load_video(video_path,end = split_ratio_,rand=args.rand,zero =args.zero,num_frm=50)
                        ans1 = [correct_ans,"Yes"]
                        video_frames_2 = load_video(video_path,start = split_ratio_,rand=args.rand,zero = args.zero,num_frm = 50)
                        #ans2 = correct_ans
                        video_frames_1.extend(video_frames_2)
                    
                    elif (args.int_sparse) and (args.unint_sparse):
                        video_frames_1 = load_video(video_path,end = split_ratio_,rand=args.rand,zero =args.zero,num_frm=10)
                        ans1 = [correct_ans,"Yes"]
                        video_frames_2 = load_video(video_path,start = split_ratio_,rand=args.rand,zero = args.zero,num_frm =10)
                        #ans2 = correct_ans
                        video_frames_1.extend(video_frames_2)
                    elif not(args.int_sparse) and args.unint_sparse:
                        video_frames_1 = load_video(video_path,end = split_ratio_,rand=args.rand,zero =args.zero,num_frm=90)
                        ans1 = [correct_ans,"Yes"]
                        video_frames_2 = load_video(video_path,start = split_ratio_,rand=args.rand,zero = args.zero,num_frm = 10)
                        #ans2 = correct_ans
                        video_frames_1.extend(video_frames_2)
                    
                    
                        #print(video_frames_1)
                        


                        
        else:
            d = 1
         
            

        #question= "Describe why the action taking place in the video represents a crime being performed. NOTE: The output has to be limited to 10 words. "
        #print(len(video_frames_1))
        # Run inference on the video and add the output to the list
        #video_path = an[cnt-1]
        #print(video_path,"###############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #video_name_ = an[cnt-1]
        possible_options = []
        cnt_1 = 0
        output1 = ""
        flag = False
        while len(possible_options)<0:
            cnt_1+=1
            question_1 = "Take a deep breadth. Summarize the actions performed by the actor in the video and infer the list of objects associated with the actor, from the relevant visual context to the activity occurring in the video. Do not include objects and events in the summary for which there is no visual evidence.FOCUS ON THE ACTIONS TAKEN BY THE ACTOR IN THE VIDEO "
            quest_1 = "Take a deep breadth and exhaustively describe all the activities that are possibly occuring in the given video"
                        
            output1 = video_chatgpt_infer(video_frames_1, quest_1, conv_mode, model, vision_tower,
                                                         tokenizer, image_processor, video_token_len)

            lt = output1.split(' ')
           # 
            if len(lt)>=1000 or "not clear" in output1 or "not possible" in output1:
                flag = False
            else:
                flag = True
                
            if cnt_1>9:
                possible_options.append(output1)
            if flag:
                possible_options.append(output1)
        flag = True
        for i in range(3):
            poss_options = ""
            random.shuffle(possible_options)#.shuffle()
            for i in range(0):
                poss_options+=chr(65+i)+": "+ possible_options[i]
            print(poss_options)
        #
            sample_set['pred'] = []
        
            flag = True
            while flag:
                q = f"The list of possible descriptions for the video are given as {poss_options}. Select the most appropriate description for the given video.".format(poss_options)
                ouptput1 = video_chatgpt_infer(video_path, q, conv_mode, model, vision_tower,
                                                     tokenizer, image_processor, video_token_len)
                poss_op = []
                while len(poss_op)<3:
                    quest_1= f"Your are provided the description of video {output1}. Logically infer the most probable intention of the actions being attempted in this video.  NOTE: THE RESPONSE MUST NOT EXCEED 10 WORDS."
                    output1 = video_chatgpt_infer(video_frames_1, quest_1, conv_mode, model, vision_tower,
                                                             tokenizer, image_processor, video_token_len)

                    print("CRIME ",output1)
                    wordsnas = output1.split(' ')
                        
                    if len(wordsnas)<150 and "does not" not in output1 and "not possible" not in output1 and "intention" in output1:
                        poss_op.append(output1)
                        flag = False
                    else:
                        flag = True
                
                q = f"The list of possible goals for the video are given as {poss_op}. Select the most appropriate goal of the action occurring in the given video."
                output1 = video_chatgpt_infer(video_frames_1, q, conv_mode, model, vision_tower,tokenizer, image_processor, video_token_len)
                poss_op = []
                while len(poss_op)<3:
                    quest_1= f"The goal of the intended activity taking place in the given video is described as: {output1}, provide a visual description of the event that leads to the failure to perform the activity with the greatest probability.  NOTE: THE RESPONSE MUST NOT EXCEED 10 WORDS."
                    output1 = video_chatgpt_infer(video_frames_1, quest_1, conv_mode, model, vision_tower,
                                                             tokenizer, image_processor, video_token_len)

                    wordsnas = output1.split(' ')
                        
                    if len(wordsnas)<150 and "does not" not in output1 and "not possible" not in output1 and "failure" in output1:
                        poss_op.append(output1)
                        flag = False
                    else:
                        flag = True
                
                
                
                q = f"The list of possible reasons behind failure of the action the video are given as {poss_options}. Select the most appropriate reason behind the failure to achieve the goal for the given video."
                output4 = video_chatgpt_infer(video_frames_1, q, conv_mode, model, vision_tower,tokenizer, image_processor, video_token_len)
                wordsnas = output4.split(' ')
                if len(wordsnas)<200 and  "does not" not in output4 and "not possible" not in output4 :
                    flag = False
                else:
                    flag = True
                print("ANS",output4)    

        
        
        
                output = main(video_path, question)
                sample_set['video_name'] = video_name_
                sample_set['Q'] = question
                sample_set['pred'].append(output)
                sample_set['G'] = goal
                sample_set['ans'] = ans1
                sample_set['actual_wrong'] = actual_wrong
                sample_set['actual_options'] = ans
                output_list.append(sample_set)

                #sample_set = {}
                #sample_set['pred']  = []
            
        else:
            d=1
        samples_set = {}
        #print(output_list)
        
            #ept Exception as e:
        #    print(f"Error processing video file '{video_name}': {e}")
           # print(sample_set)
       
    # Save the output list to a JSON file
        if cnt%1==0:
            with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
                json.dump(output_list, file)


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)