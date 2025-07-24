
# %%
# Import Modules

from __future__ import generators

# if using original HDM-HRR code: 
from hdm_hrr_original import HRR, HDM, Finst

# from hdm_hrr import HRR, HDM, Finst
import os
import python_actr
import random
from word_list import word_associations
from python_actr import (ACTR, Model, Buffer, Memory, DMNoise, DMBaseLevel)
from typing_extensions import TypeVarTuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,ifft
from numpy.linalg import norm
from nltk.corpus import stopwords
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# Run these lines ONCE if you haven't already:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# %%
# END ND 0 No matches found above threshold of cosine
log=python_actr.log(screen=False, html=False)

# RUN MODEL

# Model Setting Controls

# Set address for environment and memory vectors 
# Format: Python dictionary with words as key and Plate (1995) HRR objects as values
# Note: memory vectors contain semantic information from distributional semantics
#       environment vectors should contain random permutations of memory vector values as env values

# filename = 'lsa_semantic_memory.pkl'
# env_file = 'lsa_env_vectors.pkl'

# If using Original HDM/HRR code vectors: 
filename = 'lsa_ogHRR_semantic_memory.pkl'
env_file = 'lsa_ogHRR_env_vectors.pkl'

# Set size and source of pre-experiment semantic memory
# Note: generates a constrained semantic memory containing the words closest to each item of the experiment 

semantic_memory_size = 1000 # if None, use the full semantic memory

# ---Semantic Weight Parameter---
# Parameter for scaling the strength of base semantic vectors; strengthens the weight of initial semantic information within a composite memory chunk 
#       Semantic weight adjusts the intial signal strength of the semantic representation memory vector.
#       i.e. the memory vector of a particular word, mword, initialises with a convolved vector, mword = semantic_weight*semantic_vector 
#       This parameter can make the semantic part of a composite chunk more acessible when context vectors are later added in 
#       mword = (semantic_weight*semantic_vector) + context_vector
#       ***Default: A semantic_weight of 1.0 maintains the original magnitude of the semantic vectors.
semantic_weight = 1.0

# ---Semantic Noise Parameter---
# Parameter for scaling the strength of base semantic vectors; adds one instance of noise to each memory vector in pre-experiment memory
#       Items not accessed during experimental study do not decay due to the fixed nature of long term semantic memory 
#       Furthermore, since context vectors are not added to the compsite trace of unaccessed memories
#       Semantic chunks may be highly sensitive to similarly formatted recognition requests 
#       Memory is initialized with a random noise vector superimposed on each memory vector
#       This parameter scales the random noise vector, allowing for direct control over cos similarity 
#       Can be thought of as "pushing" these items further back in memory 
#       i.e. the memory vector of a particular word, mword, initialises as mword = semantic_vector + (semantic_noise_parameter*random_vector)
#       Note: functions similarly to the noise parameter in HDM 
#       ***Default: ***Default: A semantic_noise_parameter of 0.0 maintains the original magnitude of the semantic vectors.
#                  Vectors in the model are normalised, therefore a semantic noise parameter above 1.0 is not recommended. 
#                  i.e. semantic noise parameter of 2.0 means that the noise vector is twice as strong as the semantic vector 
semantic_noise_parameter = 0.4

# ---Semantic Noise Layers Parameter---
# Parameter for controling the number of random noise vectors added into pre-experiment semantic memory 
semantic_noise_layers = 20

# Stimulus settings 
# Set the number and length of stimulus lists presented to participants
# Experimental lists should be a dictionary containing a critical lure as key and a list of most similar words as values
# Note: Default experimental lists taken from Roediger and Mcdermott (1995)
#       List length and number of lists should not be set to a value more than the constraints of experimental_lists 

experimental_lists = word_associations
number_of_lists = 6           #number of lists for recall
number_of_simulations = 36     #number of participants
words_per_list = 12            #number of words per list 
# 6 lists of 12  words default

# BEAGLE Memory 
def load_pickle(filename):
    """Loads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:  # 'rb' for read binary
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except pickle.UnpicklingError:
        print(f"Error: Invalid pickle file '{filename}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

semantic_memory_original = load_pickle(filename)
semantic_memory_env_original = load_pickle(env_file)

def constrain_semantic_memory(memory_dict, experimental_words, n=None):
    """
    Creates a new dictionary of size n containing the most similar HRR vectors
    to the vectors of the experimental words using matrix operations.

    Args:
        memory_dict (dict): Dictionary where keys are words and values are HRR vectors (NumPy arrays).
        experimental_words (dict): List of words to find similar vectors for. Critical lure is key while related lists are value
        n (int): The desired size of the new dictionary.

    Returns:
        dict: A new dictionary containing the n most similar word-HRR pairs.
    """
    if not experimental_words or not memory_dict:
        return {}
    
    if n is None:  
        return memory_dict

    memory_words = list(memory_dict.keys())
    memory_vectors = np.array([hrr.v for hrr in memory_dict.values()])

    # convert experimental words to a list

    experiment_context = [f"{key}" if i == 0 else item for key, value in experimental_words.items() for i, item in enumerate([key] + value)]

    experimental_indices = [i for i, word in enumerate(memory_words) if word in experiment_context]
    if not experimental_indices:
        return {}

    experimental_vectors = memory_vectors[experimental_indices]
    #print(f'length of experimental indices should be length of experimental words:{len(experiment_context)}, actual length: {len(experimental_indices)}')
    experimental_lookup = {word: memory_dict[word] for word in experiment_context if word in memory_dict}

    # Calculate cosine similarity between experimental words and all memory words
    similarity_matrix = cosine_similarity(experimental_vectors, memory_vectors)

    similar_words_with_scores = []
    num_exp_words = len(experimental_lookup) #number of words  used to prepare experimental lists
    num_per_exp = (n // num_exp_words) + 1 if num_exp_words > 0 else 0

    for i, exp_word in enumerate(experimental_lookup):
        row_similarities = similarity_matrix[i]
        # Get indices of most similar words (excluding the experimental word itself)
        similar_indices = np.argsort(row_similarities)[::-1]
        count = 0
        for idx in similar_indices:
            similar_word = memory_words[idx]
            if similar_word != exp_word:
                similar_words_with_scores.append((row_similarities[idx], similar_word, memory_vectors[idx]))
                count += 1
                if count >= num_per_exp:
                    break

    # Sort all similar words globally by similarity score
    global_sorted_similar = sorted(similar_words_with_scores, key=lambda item: item[0], reverse=True)

    # Select the top n unique words
    top_n_dict = {}
    count = 0
    for score, word, vector in global_sorted_similar:
        if word not in top_n_dict:
            top_n_dict[word] = HRR(data=vector)
            count += 1
            if count >= n:
                break

    # Ensure experimental words are included if not already present
    for word in experimental_lookup:
        if word not in top_n_dict and count < n:
            top_n_dict[word] = experimental_lookup[word]
            #top_n_dict[word] = HRR(data=experimental_lookup[word])
            count += 1
            if count >= n:
                break

    return top_n_dict

def constrain_env(env_dict, semantic_memory):
    #return the environment dictionary but constrained to the same size as the semantic memory
    memory_keys = set(semantic_memory.keys())
    env_keys = set(env_dict.keys())

    # Create a new dictionary with keys from memory_dict, taking values from env_dict
    updated_env = {key: env_dict[key] for key in memory_keys if key in env_keys}
    return updated_env

# Generate constrained semantic memory and environment

# Scale size of semantic memory 
semantic_memory = constrain_semantic_memory(semantic_memory_original, experimental_lists, semantic_memory_size)
semantic_memory_env = constrain_env(semantic_memory_env_original, semantic_memory)

#print('?' in semantic_memory_env)

def create_stimulus(input_dict, num_lists=5, words_per_list=12):
    """
    Selects a specified number of random study lists and their corresponding critical lures.
    
    Parameters:
        input_dict (dict): A dictionary where keys are critical lures and values are lists of words.
        num_lists (int): Number of study lists to sample.
        words_per_list (int): Number of words to retain from each study list.
    
    Returns:
        tuple: A list of truncated study lists and a list of their corresponding critical lures.
    """
    # Ensure num_lists does not exceed available lists
    num_lists = min(num_lists, len(input_dict))
    
    # Randomly sample critical lures
    critical_lures = random.sample(list(input_dict.keys()), num_lists)
    
    # Get the corresponding study lists
    study_lists = [input_dict[key][:words_per_list] for key in critical_lures]
    full_study_lists = [input_dict[key] for key in critical_lures]
    # Check if words_per_list exceeds any list length
    max_list_length = max(len(input_dict[key]) for key in critical_lures)
    if words_per_list > max_list_length:
        print(f"Warning: words_per_list ({words_per_list}) exceeds the length of at least one list ({max_list_length}). Truncating to available length.")
    
    return study_lists, critical_lures, full_study_lists

def create_recognition_test(stimulus_lists, critical_lures, word_associations):
    # Initialize the lists that will hold the test words
    # note: stimulus list is assumed to be in the format of DRM study lists, where the item at
    # the start of a list is the most highly associated word to a critical lure 
    # return two lists, one containing the recgnition test and another containing the presented/studied items

    study_words = []
    non_presented_words = []
    non_presented_lures = []
    weakly_related_lures = []
    
    # Loop through the presented stimulus lists
    
    for stim_list in stimulus_lists:
        #index and value of stimulus_lists
        # Get study words from positions 1, 8, and 10 (adjusting for 0-based index)
        #study_words.extend([stim_list[i] for i in [0, 7, 9]])

        #Recogntion task construction (Pardilla-Delgado & Payne, 2017)

        #Include study words (i.e. words presented at encoding) from positions 1, 8, and 10 from each list included in the encoding task2.
        #Include all critical lure words (i.e. false words not presented at encoding that represent the gist of the word list) from each list included in the encoding task. 
        #Include the same number of additional, non-presented words (i.e. foil words that are unrelated to any of the studied DRM lists), from other, non-studied DRM word lists, from the same positions (1, 8, and 10) and their corresponding critical lures.
        #For example, if 15 DRM word lists are presented during encoding, for the recognition test, present 120 words: 45 study words (3*15), 15 critical lure words, 45 nonpresented list items from other non-studied DRM word lists, and 15 critical words from those nonstudied DRM word lists.
        
        if len(stim_list) < 3:
            # originally, get study words from positions 1, 8, and 10 (adjusting for 0-based index)
            # adjusted, code to accunt for list length, retreive a random item from the start, middle, and end of studied lists
            study_words.extend(stim_list)
        else: 
            start_index = 0
            third_index = len(stim_list) // 3
            middle_index = third_index * 2
            end_index = len(stim_list) 

            start_value = random.choice(stim_list[start_index:third_index])
            middle_value = random.choice(stim_list[third_index:middle_index])
            #if len(stim_list) > 12:
                #end_value = stim_list[end_index]
            # end_value = random.choice(stim_list[middle_index:end_index])
            # end_value = stim_list[end_index] <- ensures that weakly related items are NOT studied
            end_value = stim_list[-1]
            weakly_related_lures.append(end_value)
            familiar_words = [start_value, middle_value, end_value]
            study_words.extend(familiar_words)
    # Loop through all possible word associations and collect non-presented words, not related words
    # filter word associatins for words not studied 

    not_studied = {critical_lure: experimental_lists[critical_lure] for critical_lure in experimental_lists if critical_lure not in critical_lures}
    
    # number of not studied lists must be the same as number of lists studied 
    # len(stimulus_lists) is the number of lists presented during the experiment
    i = 0 
    selected_not_studied = [] 
    #selected_not_studied is used to prevent selecting the same not resented lists multiple times
    while i < len(stimulus_lists):
        random_critical_lure = random.choice(list(not_studied.keys()))
        if random_critical_lure not in selected_not_studied:
            selected_not_studied.append(random_critical_lure)
            not_presented_list = not_studied[random_critical_lure]
            # ensure the non-presented list is the same length as the presented list 
            # note: used the length of the first stimulus list since all lists are the same length
            associated_list = not_presented_list[:len(stimulus_lists[0])]
            if len(associated_list) < 3:
                non_presented = associated_list
            else: 
                nstudied_start_index = 0
                nstudied_third_index = len(associated_list) // 3
                nstudied_middle_index = nstudied_third_index * 2
                nstudied_end_index = len(associated_list) 

                nstudied_start_value = random.choice(associated_list[nstudied_start_index:nstudied_third_index])
                nstudied_middle_value = random.choice(associated_list[nstudied_third_index:nstudied_middle_index])
                nstudied_end_value = random.choice(associated_list[nstudied_middle_index:nstudied_end_index])

                non_presented = [nstudied_start_value, nstudied_middle_value, nstudied_end_value]
            
            non_presented_words.extend(non_presented)
            non_presented_lures.append(random_critical_lure)
            i+=1
    # Create the full test list by combining all categories
    test_list = study_words + critical_lures + non_presented_words + non_presented_lures
    
    # Shuffle the test list to randomize the order
    random.shuffle(test_list)
    return test_list, study_words, weakly_related_lures

class RecognitionModel(Model): 
    note = Model(word="START")
    display = Model(word="START")
    
    #self.report[self.listnumber] = [] 
    
    #stimulus_lists, critical_lures, full_study_lists = create_stimulus(experimental_lists, number_of_lists, words_per_list)
    #recognition_test, studied, weakly_related_lures = create_recognition_test(full_study_lists, critical_lures, experimental_lists)
    #note is the same as display in main.py 

    def __init__(self):
        super().__init__()  # Calls the parent class's __init__ method
        self.listnumber = 0  # Instance variable
        self.stimulus_lists, self.critical_lures, self.full_study_lists = create_stimulus(experimental_lists, number_of_lists, words_per_list)
        self.recognition_test, self.studied, self.weakly_related_lures = create_recognition_test(self.full_study_lists, self.critical_lures, experimental_lists)
        self.words_recognized = []
        self.words_notrecognized = []
        self.presented = []
        self.report = {}

    def get_random_list(self, lists):
        """
        Returns a random list from the given list of lists, 
        ensuring no list is repeated until all lists have been used.

        random list also keeps track of lists that have already been presented
        as such, it is kept within the cognitive model to allow for it to re-initialize with all stimulus lists 
        per participant  
        """
        if not lists:
            return None  # No lists available
        # If 'used_indices' doesn't exist, initialize it to track used indices
        if not hasattr(self.get_random_list, 'used_indices'):
            self.get_random_list.used_indices = []
        # If all lists have been used, return None
        if len(self.get_random_list.used_indices) == len(lists):
            return None  # All lists have been used
        while True:
            index = random.randint(0, len(lists) - 1)
            if index not in self.get_random_list.used_indices:
                self.get_random_list.used_indices.append(index)
                return lists[index]    

    def start_list(self): 
        stimuli = self.get_random_list(self.stimulus_lists)
        print(f"Current stimuli: {stimuli}")
        #if no more lists are available 
        if not stimuli: 
            yield 1 
            self.display.word = "NONE" 
            yield 1
            #if all lists presented, then start recognition test 
            self.display.word = "COMPLETE"
            print("All lists have been presented. Starting recognition task.")
        else:
            self.presented.append(stimuli)
            self.listnumber +=1
            print(f"The current list number is: {self.listnumber}")
            if self.listnumber not in self.report: 
                self.report[self.listnumber] = ([], []) 
            print(f"Initialized recall report for list {self.listnumber}.")

            stimuli = ["START"] + list(stimuli) + ["END", "END", "RECALL"]
            for stimulus in stimuli:
                #1 word every 2 sec.
                yield 2
                self.display.word = "NONE" 
                yield 2
                self.display.word = stimulus
            #yield 10 #ensures that full recall is completed before finish is dislayed and activation is captured 
            #activation is captured for recall when display is finish 
            #recall begins when display is recall 
            #when display word is FINISH, sub_init fires and self.listnumber increases
            #self.display.word = "FINISH"

    def report_finish(self):
        # When participant finishes recall, set display to finish and prepare for next list
        # Also sorts recall report into participant serial recall order 
        self.display.word = "FINISH" 

        # Code for sorting recall output
        if self.listnumber not in self.report:
            self.report[self.listnumber] = []
        else: 
            serial_order = self.report[self.listnumber][1] + self.report[self.listnumber][0]
            self.report[self.listnumber] = serial_order

    def start_recognition(self):
        #create a list of words from a recognition task in the order lists were presented 
        test = self.recognition_test
        print(f"Recognition test: {test}")
        test = list(test) + ["END"]
        for item in test: 
            yield 1 #spend one second reading and one second recognising 
            self.note.word = "NONE"
            yield 1 
            self.note.word = item
        self.note.word = "FINISH"

    def report_word(self, word, reverse_recall=False): 
        #report word; cannot be off, START, END, experiment (contextual tags)
        if self.listnumber not in self.report: 
            self.report[self.listnumber] = ([],[]) # (reverse_list, forward_list)
        reverse_list = self.report[self.listnumber][0]
        forward_list = self.report[self.listnumber][1]
        
        # Filter out non-words and special markers first
        if word in ["START", "END", "off", "EXPERIMENT"]:
            return
        
        # Corrected duplicate check: Check if the word is in EITHER the reverse_list OR the forward_list
        if word in reverse_list or word in forward_list:
            return # Word already reported, so skip it
        
        if reverse_recall:
            reverse_list.insert(0, word)
        else: 
            forward_list.append(word)
    
    def report_none(self):
        #if no words are recalled, identify this in the end report 
        if self.listnumber not in self.report: 
            self.report[self.listnumber] = [] 

    #functions for recognition test response 

    def sayYes(self,word): 
        if word != "START":
            if word != "END":
                self.words_recognized.append(word)
    
    def sayNo(self,word): 
        if word != "START":
            if word != "END":
                    self.words_notrecognized.append(word)

    # turn off the display at the end of a list presentation 
    def end_recognition(self):
        self.note.word = "OFF"

    def display_stop(self): 
        self.display.word = "OFF"

    
#Create ACT-R Agent 
class ListMemorizerSingle(ACTR):
    focus = Buffer()
    auditory = Buffer()
    DMbuffer = Buffer()
    list_buffer = Buffer()
    query_buffer = Buffer()
    
   
    # PARAMETERS
    Use_HDM = True
    if Use_HDM: 
        DM = HDM(DMbuffer, N=300, latency=0.05, noise=0.03,forgetting=1.0, threshold=-2.6, maximum_time=10.0, verbose=False, finst_size=30,finst_time=200.0, use_experimental_noise=True)
    
    # standard latency = 0.05

    # Threshold -4 -> cosine 0.134
    #forgetting coefficient α ranges from 0 to 1
    #If α = 0, the model has complete amnesia, and if α = 1, the model does not privilege more recent information over older information.
    #>1 values older information more thann recent information 
    #finst prevents things that have already been recalled from being recalled again, the size is larges enough for both forward and reverse search  
    #threshold and latency taken from HDM example simple spread
    #add noise and forgetting accordingly
    # DM=HDM(DMbuffer,N=64,latency=0.5,verbose=False,noise=1.0,forgetting=0.9,finst_size=22,finst_time=100.0)
    else: 
        DM=Memory(DMbuffer,latency=3.0,threshold=0,finst_size=22,finst_time=100.0)
                                                     # latency controls the relationship between activation and recall
                                                     # activation must be above threshold - can be set to none

        dm_n=DMNoise(DM,noise=0.0,baseNoise=0.0)         # turn on for DM subsymbolic processing
        dm_bl=DMBaseLevel(DM,decay=0.5,limit=None)       # turn on for DM subsymbolic processing
    
    # Productions for initiating model 

    # Fill environment with pre-experiment word-HRR pairs representing semantic relationships 

    def LoadFile(self): 
        # 1. Populate model environment with semantic vectors 
        self.DM.env.update(semantic_memory_env)
        print(f'? is in semantic_memory_env:{"?" in semantic_memory_env}')
        # 2. Add items to model memory as self-associative semantic chunks 
        
        # Identify words in the environment 
        preexperiment_items = list(self.DM.env.keys())
        # print(f'? is in pre_experiiment_items:{"?" in preexperiment_items}')

        # Add words to model memory 
        for word_item in preexperiment_items:
            if word_item == '?':
                print(f"Skipping adding the placeholder vector named '{word_item}' to DM.mem.")
                continue # Skip this item
            self.DM.add(f'following_word:{word_item} value:{word_item} preceding_word:{word_item}', use_exp_memory=False)
        
        # 3. Apply semantic weighting, semantic noise and semantic layers to memory  
        
        for word in self.DM.mem:
            # Create a random semantic noise vector accumulator 
            combined_noise_vector = HRR(N=self.DM.N, zero=True) # Initialize a zero vector
            if semantic_noise_parameter > 0:
                # Loop to generate and sum 'semantic_noise_layers' number of noise vectors
                for layer in range(semantic_noise_layers):
                    # Generate a new random vector for each layer
                    new_noise_layer = HRR(N=self.DM.N) * semantic_noise_parameter
                    combined_noise_vector += new_noise_layer
            # Scale semantic component of memory vector and add the combined noise
            if semantic_weight == 0:
                print(f'WARNING: Semantic vector for {word} initialized as zero vector, with noise.')
                self.DM.mem[word] = combined_noise_vector
            else:
                self.DM.mem[word] = (self.DM.mem[word] * semantic_weight) + combined_noise_vector

    # Initiate Buffers and list presentation 

    def init():
        self.LoadFile()
        print("memory loaded...")
        auditory.set('START START START')
        query_buffer.set('END')
        focus.set('setlist')
        self.parent.start_list()
    
    # Production for initiating list presentation after the initial list

    def sub_init(focus='phase:RECALL waiting:NEXT'): 
        #present another list following the first
        self.parent.display_stop() 
        self.parent.start_list()
        auditory.set('START START START')
        query_buffer.set('END')
        focus.set('setlist')

    # Production for holding current list context 

    def set_listnumber(focus='setlist'): 
        list_buffer.set(f'context:EXPERIMENT list_number:{self.parent.listnumber}')
        focus.set('phase:STUDY waiting:TRUE rehearsal:FALSE')
    
    # Production for initiating Recognition Test after all lists have been presented 

    def start_recognition(focus='phase:STUDY waiting:TRUE', display='word:COMPLETE'): 
        #when display word is finish, the recognition test begins, switch display to off afterwards so this production does not continuously fire
        self.parent.start_recognition()
        auditory.set('START START')
        focus.set('phase:RECOGNITION waiting:TRUE rehearsal:FALSE')  

    # Productions for keeping track of the present word during list presentation

    # When display word is NONE, prepare for the upcoming word 

    def prepare_for_word(focus='phase:STUDY waiting:FALSE rehearsal:?STATUS', display='word:NONE'):
        focus.set('phase:STUDY waiting:TRUE rehearsal:?STATUS')
    
    # Set auditory buffer to the current being presented word

    def notice_word(focus='phase:STUDY waiting:TRUE', display='word:?A!NONE!COMPLETE!FINISH!RECALL!OFF', auditory='?B ?C ?D'):
        # Initially auditory buffer holds START START START 
        # After word1 is presented, auditory buffer becomes word1 START START
        # After word2..., auditory buffer becomes word2 word1 START
        print("I notice the new word is " + A)
        auditory.set('?A ?B ?C')
        focus.set('phase:STUDY waiting:FALSE rehearsal:FALSE')   
    
    # Productions for rehearsal 

    # Production for rehearsing the current item, only beings firing when current item is not START

    def basic_rehearse(focus='phase:STUDY', auditory='?A!START ?B ?C', list_buffer='context:EXPERIMENT list_number:?number'):
        # Encode the current item into memory 
        # structure: following_word:word2 value:word1 preceding_word:START

        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 

        # Encode Contextual Chunk in each Serial Position
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        #DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)  

        # Encode Semantic Chunk
        DM.add('following_word:?A value:?A preceding_word:?A', use_exp_memory=True) 
        # DM.add('following_word:?A value:?A preceding_word:?A') 

    # Productions for rehearsing the first item presented

    def rehearse_add(focus='phase:STUDY waiting:?STATUS rehearsal:FALSE', auditory='?A!START ?B!START ?C!START', list_buffer='context:?context list_number:?number'):
        # Encode current item 

        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 

        # Encode Contextual Chunk in each Serial Position 
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        
        #DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)  

        # Encode Semantic Chunk
        DM.add('following_word:?A value:?A preceding_word:?A', use_exp_memory=True) 

        DM.request('following_word:? value:START preceding_word:START list_number:?number context:EXPERIMENT')
        focus.set('phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARD')

    def rehearse_retrieve(focus='phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARD', DMbuffer='following_word:?A value:?B preceding_word:?C list_number:?number context:EXPERIMENT', DM='busy:False'):
        # DMBuffer now contains the first item in the present list 
        # All lists are encoded with identical START and END markers, using the list slot ensures list distinction so the first item of the current list is retrieved 
        
        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 

        # Encode Contextual Order Chunk
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        #DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)  

        # Encode Semantic Chunk
        DM.add('following_word:?A value:?A preceding_word:?A', use_exp_memory=True) 

        # Request the item after the first item
        DM.request('following_word:? value:?A preceding_word:?B') 
        focus.set('phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARDCHAIN')
    
    def rehearse_retrieve_chain(focus='phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARDCHAIN', DMbuffer='following_word:?A value:?B preceding_word:?C', list_buffer='context:?context list_number:?number', DM='busy:False'): 
        # rehearse using only associative information, further recall from this production does not employ the list context
        
        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True)
        #DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
 
        # Encode Contextual Chunk
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        #DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)  

        # Encode Semantic Chunk
        DM.add('following_word:?A value:?A preceding_word:?A', use_exp_memory=True) 

        # Continue chain recall until no longer possible 
        focus.set('phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARDCHAIN')

    def rehearse_forget(focus='phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARD',DMbuffer=None, DM='error:True'):
        # This forgetting production fires if rehearse_add fails and the first item cannot be recalled
        focus.set('phase:STUDY waiting:?STATUS rehearsal:FALSE')
    
    def rehearse_forget_chain(focus='phase:STUDY waiting:?STATUS rehearsal:TRUE search:FORWARDCHAIN',DMbuffer=None, DM='error:True'):
        # This forgetting production fires if rehearse_retrieve_chain fails and the next item in the associative chain cannot be recalled
        focus.set('phase:STUDY waiting:?STATUS rehearsal:FALSE')


    #BEGIN RECALL TEST 
    # RECALL THE LIST, SEARCHING BACKWARD FROM THE END
    # begin recall when displayed word is 'RECALL' 
    # start and end variable should be unique per list ?
    # setting require_new to true means it wont recall again something just recalled, marked by finst
    #during recall, falsely recalled words are also associated witht he context as individuals believe it is from the experiment

    def begin_recall(focus='phase:STUDY waiting:TRUE', display='word:RECALL'):
        print("Trying to do free recall of the list.")
        DM.finst.obj = [] #reset finst at the end of recall to make sure items from the last list do not immediately interrupt
        #DM.request('current_word:?A previous_word:?B initial_word:? list_number:?number',require_new=True)
        focus.set('phase:TEST waiting:FALSE search:REVERSE')
    
    # Productions for recalling the END item of a list 

    def request_end(focus='phase:TEST waiting:FALSE search:REVERSE', list_buffer='context:?context list_number:?number'):
        print("Finst contains:", str(DM.finst.obj))
        # Request the end of list item using the END marker (?query): following_word:? value:END preceding_word:END
        DM.request('following_word:END value:END preceding_word:? list_number:?number context:EXPERIMENT',require_new=True)
        focus.set('phase:TEST waiting:TRUE search:REVERSE')
    
    # If end item cannot be recalled 

    def forgot_once(focus='phase:TEST waiting:TRUE search:REVERSE', list_buffer='context:EXPERIMENT list_number:?number', DMbuffer=None, DM='error:True'):
        # Forgetting production if retrieval from the start of list fails 
        print("I cannot recall the last item in the list. Let me start from the beginning.")
        self.parent.report_none()
        # Request beginning of the list
        DM.request('following_word:? value:START preceding_word:START list_number:?number context:EXPERIMENT')
        focus.set('phase:TEST waiting:TRUE search:FORWARD')

    # Else, continue chain recall guided solely by order information 
        
    def recall_end(focus='phase:TEST waiting:TRUE search:REVERSE', DMbuffer='following_word:?A value:?B preceding_word:?C list_number:?number context:EXPERIMENT', DM='error:False busy:False'):
        # Report the item recalled from the end of the list 
        print("I recall that the last item in the list was", C)
        self.parent.report_word(C, True)

        # Encode item into memory 

        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', True) 

        # Encode Contextual Chunk
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        #DM.add('value:?A context:EXPERIMENT list_number:?number', True)  

        # Encode Semantic Chunk
        DM.add('following_word:?C value:?C preceding_word:?C', use_exp_memory=True) 
        
        # Request the second item in the chain; without using the list context (i.e. following_word:END value:LastWord preceding_word:? )

        auditory.set('?B ?C')

        # Initiate recall production
        focus.set('phase:TEST waiting:FALSE search:REVERSECHAIN')
    
    
    # Productions for chain recalling the rest of the list after the end item is retrieved 

    def request_reverse(focus='phase:TEST waiting:FALSE search:REVERSECHAIN', auditory='?B ?C'):
        print("Finst contains:", str(DM.finst.obj))
        DM.request('following_word:?B value:?C preceding_word:?',require_new=True)
        focus.set('phase:TEST waiting:TRUE search:REVERSECHAIN')
        
    def recall_reverse(focus='phase:TEST waiting:TRUE search:REVERSECHAIN', DMbuffer='following_word:?A value:?B preceding_word:?C!START', list_buffer='context:?context list_number:?number',DM='error:False busy:False'):
        # Report the item recalled from the end of the list 
        print("I recall ", C)
        self.parent.report_word(C, True)

        # Encode item into memory 

        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', True)

        # Encode Contextual Chunk
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?C context:EXPERIMENT list_number:?number', use_exp_memory=True) 
        #DM.add('value:?A context:EXPERIMENT list_number:?number', True) 

        # Encode Semantic Chunk
        DM.add('following_word:?C value:?C preceding_word:?C', True) 

        # Semantic encoding 
        # DM.add('following_word:?C value:?C preceding_word:?C') 
        #DM.add('following_word:?C value:?C preceding_word:?C') 

        # Request the next item in the chain if possible
        auditory.set('?B ?C')

        focus.set('phase:TEST waiting:FALSE search:REVERSECHAIN')
    
    def found_start(focus='phase:TEST waiting:TRUE search:REVERSECHAIN', DMbuffer='following_word:?A value:?B preceding_word:START', list_buffer='context:?context list_number:?number', DM='error:False busy:False'):
        print("I recalled all words up till the start of the list. Let's search forward now.")
        #auditory.set('START')
        DM.request('following_word:? value:START preceding_word:START list_number:?number context:EXPERIMENT')
        focus.set('phase:TEST waiting:TRUE search:FORWARD')

    def forgot_once_chain(focus='phase:TEST waiting:TRUE search:REVERSECHAIN', DMbuffer=None, DM='error:True', list_buffer='context:?context list_number:?number'):
        # Forgetting production if chain retrieval fails
        print("I forget. Let me start from the beginning.")
        self.parent.report_none()
        #auditory.set('START')
        DM.request('following_word:? value:START preceding_word:START list_number:?number context:EXPERIMENT')
        focus.set('phase:TEST waiting:TRUE search:FORWARD')

    
    # Productions for forward recall

    def request_forward_start(focus='phase:TEST waiting:FALSE search:FORWARD', DMbuffer ='following_word:?A value:?B preceding_word:?C list_number:?number context:EXPERIMENT', DM='error:False busy:False'):
        # Initially DMbuffer is current_word:?A previous_word:START initial_word:START 
        print("Finst contains:", str(DM.finst.obj))

        print("I recall the first word is ", A)
        self.parent.report_word(A)

        # Encode START item into memory 

        # Context encoding START marker
        #DM.add('following_word:?A value:START preceding_word:START list_number:?number context:EXPERIMENT')
        #DM.add('following_word:?A value:START preceding_word:START list_number:?number context:EXPERIMENT')
        #DM.add('following_word:?A value:START preceding_word:START list_number:?number context:EXPERIMENT') 

        # Semantic encoding 
        #DM.add('following_word:?A value:?A preceding_word:?A') 

        # Encode Serial Order Chunk
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', True) 

        # Encode Contextual Chunk 
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory= True) 
        #DM.add('value:?A context:EXPERIMENT list_number:?number', True) 

        # Encode Semantic Chunk
        DM.add('following_word:?A value:?A preceding_word:?A', use_exp_memory=True) 

        # Request the next item in the forward chain

        auditory.set('?A ?B')

        focus.set('phase:TEST waiting:FALSE search:FORWARDCHAIN')
    
    # Production for forward chain recall without list context 

    def request_forward(focus='phase:TEST waiting:FALSE search:FORWARDCHAIN', auditory='?A ?B'):
        # Initially DMbuffer is current_word:?A previous_word:START initial_word:START 
        
        # Request the next item in the forward chain

        DM.request('following_word:? value:?A preceding_word:?B',require_new=True) 
        
        focus.set('phase:TEST waiting:TRUE search:FORWARD')

    def recall_forward(focus='phase:TEST waiting:TRUE search:FORWARD', DMbuffer='following_word:?A!END!START value:?B preceding_word:?C', list_buffer='context:?context list_number:?number', DM='error:False busy:False'):  
        print("Finst contains:", str(DM.finst.obj))

        print("I recall", A)
        self.parent.report_word(A)

        # Encode Serial Order Chunk 
        DM.add('following_word:?A value:?B preceding_word:?C', use_exp_memory=True) 
        #DM.add('following_word:?A value:?B preceding_word:?C', True) 

        # Encode Context Chunk 
        DM.add('following_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('preceding_word:?A context:EXPERIMENT list_number:?number', use_exp_memory=True)
        DM.add('value:?A context:EXPERIMENT list_number:?number', use_exp_memory=True) 
        #DM.add('value:?A context:EXPERIMENT list_number:?number', True) 

        # Encode Semantic Chunk
        DM.add('following_word:?A value:?A preceding_word:?A', use_exp_memory=True) 

        auditory.set('?A ?B')
        focus.set('phase:TEST waiting:FALSE search:FORWARDCHAIN')

    def forgot_twice(focus='phase:TEST waiting:TRUE search:FORWARD', DMbuffer=None, DM='error:True'):
        print("I forgot again. Let's quit")
        self.parent.report_finish()
        focus.set('stop')

    def found_end(focus='phase:TEST waiting:TRUE search:FORWARD', DMbuffer='following_word:END value:?B preceding_word:?C', DM='error:False busy:False'):
        print("I'm at the end of the list. Let's quit.")
        self.parent.report_finish()
        focus.set('stop')

    def prepare_for_nextlist(focus='stop', display='word:FINISH'): 
        print("I'm now waiting for the investigator to present the next list")
        focus.set("phase:RECALL waiting:NEXT")

    #START RECOGNITION TEST
    
    def prepare_for_recognition(focus='phase:RECOGNITION waiting:FALSE rehearsal:?STATUS', note='word:NONE'):
        focus.set('phase:RECOGNITION waiting:TRUE rehearsal:?STATUS')
    
    def notice_recognition(focus='phase:RECOGNITION waiting:TRUE', note='word:?A!NONE!START!note!FINISH', auditory='?B ?C'):
        print("I notice the new word is " + A)
        auditory.set('?A ?B')
        focus.set('phase:RECOGNITIONTEST waiting:FALSE mode:CHECK')

    def recognition_check(focus='phase:RECOGNITIONTEST waiting:FALSE mode:CHECK', auditory='?A!END ?B'):
        print("Checking if I recognize the word " + A)
        DM.request('value:?A preceding_word:?A following_word:?A context:EXPERIMENT')
        focus.set('phase:RECOGNITIONTEST waiting:FALSE rehearsal:FALSE')
    
    # Check if word can be found in an experimental list, if not check other positional slots
    def respond_yes(focus='phase:RECOGNITIONTEST waiting:FALSE rehearsal:FALSE', DMbuffer='value:?A preceding_word:?A following_word:?A context:EXPERIMENT', DM='busy:False'):
        self.parent.sayYes(A)
        print("I recognize", A)
        focus.set('phase:RECOGNITION waiting:FALSE rehearsal:FALSE')
 
    def respond_no(focus='phase:RECOGNITIONTEST waiting:FALSE rehearsal:FALSE', DMbuffer=None, DM='error:True', auditory='?A ?B'):
        self.parent.sayNo(A)
        print('I do not recognize this word')
        focus.set('phase:RECOGNITION waiting:FALSE rehearsal:FALSE')
        
    def end_drm_experiment(note='word:FINISH'): 
        self.parent.end_recognition()
        #self.note.word is set to 'off' 
        self.stop()

    

#number of simulations/participants must be >= 1
simulation_counter = 0   

#ordered list of lists representing words recognised
participant_recog = [] 
participant_norecog = [] 


#keys are numbers representing the order of list presentation
#keys contain an ordered list of lists, the index representing a particular participant 
# participant_recall = {list1: [['car,'apple'], ['fries', 'burger']] list2: [['lemon','tree'], ['cat', 'dog']]}
participant_recall = {}
participant_recall_activation = {}
chunks_per_participant = {}
words_presented = {} 
# list of lists containing critical lures presented to each participant
critical_lures_perparticipant = []
weakly_related_lures_perparticipant = [] 

# Dictionary of lists containing participant id as key and given recognition test as value
# Index represents participant number 
recognition_tests = []

while simulation_counter < number_of_simulations: 
    participant=ListMemorizerSingle()                 # name the agent
    experiment=RecognitionModel()                     # name the environment
    experiment.agent=participant                      # put the agent in the environment                   
    python_actr.log_everything(experiment)                 # print out what happens in the environment
    #log=python_actr.log(screen=False, html=True)
    #log=python_actr.log(screen=False, html=True)

    # Before running the experiment
    
    # Alternatively, reduce the trace verbosity
    #python_actr.log.trace_level = 0
    
    #python_actr.log(False) # Disables logging entirely

    
    times = []
    finish_captured = False 
    finished_process = False 
    last_processed_time = None  # Track when results were last processed
    cooldown_period = 2.0 #2 seconds
    
  

    #finish -> self.parent.display_stop()  OFF -> list +=1 
    while experiment.note.word != "OFF":
        experiment.run(limit=0.1) # run with a limit inside a loop means run through the experiment at 0.5ms per iteration
        #make sure increments are kept low, experiment.note.word switches from  finish to off immediately so its possible to miss the production
        
        # Record Recognition Test per participant (simulation) 
        # "FINISH" is the last item in note prior to "OFF" which ends the model run
        # if conditions ensure, recognition test is captured only once per simulation 
        if experiment.note.word == "FINISH":
            if experiment.recognition_test not in recognition_tests:
                recognition_tests.append(experiment.recognition_test)

        if experiment.listnumber not in participant_recall_activation and experiment.listnumber != 0:
            participant_recall_activation[experiment.listnumber] = []
            participant_recall[experiment.listnumber] = []
            chunks_per_participant[experiment.listnumber] = []
            words_presented[experiment.listnumber] = []
        
        
        times.append(experiment.now()) # capture time of activation recording
        # if experiment.display.word == "FINISH" and (not prev_stimulus) and (not finished_process):
        if experiment.display.word == "FINISH" and (not finished_process):
            current_time = experiment.now()  # Get the current experiment time
            if last_processed_time is None or (current_time - last_processed_time > cooldown_period):
                if experiment.presented:
                    
                    #initial_stimulus = False
                    #prev_stimulus = False

                    initial_stimulus = "START"
                    prev_stimulus = "START"
                    
                    activation = [] 
                    # items presented 
                    items = experiment.presented[experiment.listnumber-1]

                    # items recalled 
                    print(f"Experiment Report: {experiment.report}")
                    recalled = experiment.report[experiment.listnumber]

                    # identify activation of each item presented immediately after recall
                    stimuli = ['START'] + items + ['END', 'END'] 

                    # list containing individual chunk triplets which represent items in the stimuli 
                    chunks = []
                    for i, stimulus in enumerate(stimuli): # Use enumerate to track position if needed

                        # Only create a chunk if we have enough words to form a triplet
                        # i.e., at least the second word (index 1) if initial_stimulus is first word,
                        # or the third word (index 2) if initial_stimulus is the word before prev_stimulus
                        if i >= 1: # We have at least 'START', prev_stimulus (first word), current stimulus (second word)
                            if i == 1: # First actual triplet, prev_stimulus is the first real word
                                # This handles (START, WORD1, WORD2)
                                initial = 'START'
                                prv = str(prev_stimulus) # This would be WORD1 from last iteration's update
                                nxt = str(stimulus) # This is WORD2
                            else: # Regular triplet: (WORD_n-2, WORD_n-1, WORD_n)
                                initial = str(initial_stimulus) # This would be WORD_n-2 from last iteration's update
                                prv = str(prev_stimulus) # This would be WORD_n-1 from last iteration's update
                                nxt = str(stimulus) # This is WORD_n

                            the_chunk = f'preceding_word:{initial} value:{prv} following_word:{nxt}'
                            a = experiment.agent.DM.get_activation(the_chunk)

                            # Apply your special START/END formatting for chunks list if needed
                            temp_initial = initial # Use temporary variables for chunk list formatting
                            temp_prv = prv
                            temp_nxt = nxt
                            if temp_initial == 'START': temp_initial = 'S'
                            if temp_prv == "START": temp_prv="S"
                            if temp_nxt=="END": temp_nxt="E" # This case will happen *after* the loop for the last chunk

                            chunks += [temp_initial+','+temp_prv+','+temp_nxt]
                            activation += [a]

                        # Update for the next iteration (THIS IS THE CRITICAL PART)
                        initial_stimulus = prev_stimulus
                        prev_stimulus = stimulus    
                    """
                    for stimulus in stimuli:
                        if prev_stimulus is not an empty string 
                        if prev_stimulus:
                            if initial_stimulus:
                                #list items are added into DM as subsequent pairs (i.e. START CHAIR)
                                #second_word:?A first_word:?B context:EXPERIMENT list_number:?number
                                the_chunk = f'preceding_word:{str(initial_stimulus)} value:{str(prev_stimulus)} following_word:{str(stimulus)}'
                                a = experiment.agent.DM.get_activation(the_chunk) 
                                initial = str(initial_stimulus)
                                prv = str(prev_stimulus)
                                nxt = str(stimulus)
                                if initial == 'START':
                                    initial = 'S'
                                if prv == "START":
                                    prv="S"
                                if nxt=="END":
                                    nxt="E"
                                chunks += [initial+','+prv+','+nxt] 
                                activation += [a]
                            initial_stimulus = prev_stimulus
                            prev_stimulus = stimulus
                        prev_stimulus = stimulus 
                    """
                    #for first participant list 1
                    if not participant_recall.get(experiment.listnumber):
                        participant_recall_activation[experiment.listnumber] = [activation]
                        participant_recall[experiment.listnumber] = [recalled] 
                        chunks_per_participant[experiment.listnumber] = [chunks] 
                        #add extra bracket here
                        words_presented[experiment.listnumber]=[experiment.presented[experiment.listnumber-1]]
                    elif simulation_counter >= len(participant_recall_activation[experiment.listnumber]): 
                        # participant_recall_activation[experiment.listnumber] is  a list of list where index is participant id 
                        # i.e. {1: [car, bread], [apple]}, for list 1, participant 1 recalled car, bread while participant 2 recalled apple
                        # simulation counter remains the same during the a single loop
                        participant_recall_activation[experiment.listnumber].append(activation)
                        #use participant_recall[experiment.listnumber].append(items) to track positinality 
                        #participant_recall[experiment.listnumber].append(recalled) for easier comparison
                        participant_recall[experiment.listnumber].append(recalled) 
                        chunks_per_participant[experiment.listnumber].append(chunks) 
                        words_presented[experiment.listnumber].append(experiment.presented[experiment.listnumber-1])
                    else: 
                        #for the same participant join new items from subsequent loops if missed 
                        #activation will always differ as it is being calculated at a different time, 
                        delete_duplicates = set(participant_recall[experiment.listnumber][simulation_counter]+recalled)
                        participant_recall[experiment.listnumber][simulation_counter] = list(delete_duplicates)
                else:
                    if experiment.listnumber not in participant_recall:
                        participant_recall_activation[experiment.listnumber] = [[]]
                        participant_recall[experiment.listnumber] = [[]]
                        chunks_per_participant[experiment.listnumber] = [[]]
                        words_presented[experiment.listnumber]=experiment.presented[experiment.listnumber-1]  
                    
                    if len(participant_recall_activation[experiment.listnumber]) != simulation_counter: 
                        participant_recall_activation[experiment.listnumber].append([])
                        participant_recall[experiment.listnumber].append([]) 
                        chunks_per_participant[experiment.listnumber].append([]) 

                prev_stimulus = False
                finish_processed = True
                last_processed_time = current_time

        if experiment.display.word != "FINISH":
            #ensures that only one instance of results are captured for each recalled list everytime experiment display is finish
            finish_processed = False

        if experiment.note.word == "FINISH" and not finish_captured:
            #FINISH is the last word presented and triggers the production which sets note to off 
            #capture results of recognition task at the end of the experiment, before experiment ends   
            participant_recog.append([*experiment.words_recognized])
            participant_norecog.append([*experiment.words_notrecognized])
            if experiment.recognition_test not in recognition_tests:
                recognition_tests.append(experiment.recognition_test)
            critical_lures_perparticipant.append(experiment.critical_lures)
            weakly_related_lures_perparticipant.append(experiment.weakly_related_lures)
            finish_captured = True
    if not finish_captured:
        print("FINISH not captured during the loop. Capturing now...")
        participant_recog.append([*experiment.words_recognized])
        participant_norecog.append([*experiment.words_notrecognized])
        critical_lures_perparticipant.append(experiment.critical_lures)
        weakly_related_lures_perparticipant.append(experiment.weakly_related_lures)
        if experiment.recognition_test not in recognition_tests:
                recognition_tests.append(experiment.recognition_test)

    simulation_counter +=1 

print(f'recognition tests: {recognition_tests}')
print(f'words recognized: {participant_recog}')
print(f'words not recognized:{participant_norecog}')
print(f'activation per list:{participant_recall_activation}')
print(f'chunks created for measuring activation in each list recall: {chunks_per_participant}')
print(f'words recalled: {participant_recall}')
print(f'words presented: {words_presented}')

python_actr.finished()                             # stop the environment

# RESULTS 

# In the original DRM task 
# Hit rate and false alarm rate are measures specifically used for recognition tests
# Proportion of intrusions (or false recall rates) are measures specifically used for recall tests

# Helper function for getting grammatical forms of a root word 

# Initialize the lemmatizer once
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def expand_word_forms(word):
    """
    Generates related grammatical forms (singular/plural, base verb/gerund) of a given word.
    Uses NLTK's WordNetLemmatizer to find base forms and adds common derived forms heuristically.
    """
    forms = {word} # Always include the original word itself
    
    # 1. Try lemmatizing as a noun (covers plurals: 'mountains' -> 'mountain')
    lemma_n = lemmatizer.lemmatize(word, pos='n')
    forms.add(lemma_n)
    # Heuristic for adding common plural form if the base noun was given
    if word == lemma_n and not word.endswith('s'): # if original is a base noun (e.g., 'mountain')
        forms.add(word + 's') # e.g., 'mountain' -> 'mountains'
        if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou': # 'city' -> 'cities'
            forms.add(word[:-1] + 'ies')
    
    # 2. Try lemmatizing as a verb (covers gerunds: 'smoking' -> 'smoke', 'running' -> 'run')
    lemma_v = lemmatizer.lemmatize(word, pos='v')
    forms.add(lemma_v)
    # Heuristic for adding common verb forms like gerunds if the base verb was given
    if word == lemma_v: # if original is a base verb (e.g., 'smoke')
        forms.add(word + 'ing') # e.g., 'smoke' -> 'smoking'
        # Handle double consonant for 'ing' (e.g., 'run' -> 'running')
        if len(word) > 1 and word[-1] in 'bcdfghjklmnpqrstvwxyz' and word[-2] in 'aeiou':
            forms.add(word + word[-1] + 'ing')
        # Handle dropping 'e' for 'ing' (e.g., 'make' -> 'making')
        if word.endswith('e') and not word.endswith('ee'):
            forms.add(word[:-1] + 'ing')
            
    return forms

def calculate_recall(words_recalled, words_presented, critical_lures_perparticipant):
    """
    Calculates the average probability of critical lure intrusions and non-critical, non-studied intrusions
    across all participants. Intrusion probabilities are calculated as the proportion of lists
    on which at least one such intrusion occurred for each participant, then averaged.

    Args:
        words_recalled (dict): A dictionary where keys are list numbers (e.g., 1, 2) and values are
                               lists of lists. Each inner list represents words recalled by a specific
                               participant for that list.
                               Example: {1: [['p1_list1_recalled'], ['p2_list1_recalled']]}
        words_presented (dict): Same structure as words_recalled, but contains words studied by participants.
        critical_lures_perparticipant (list of list): Outer list index is participant_id. Inner list contains
                                                      the critical lure for each corresponding list presented
                                                      to that participant (0-indexed by list position).
                                                      Example: [['cl_for_p1_list1', 'cl_for_p1_list2'], ['cl_for_p2_list1', 'cl_for_p2_list2']]

    Returns:
        tuple: A tuple containing:
               - avg_prob_critical_lure_intrusion (float): Average probability of recalling a critical lure.
               - avg_prob_non_critical_intrusion (float): Average probability of recalling a non-critical, non-studied intrusion.
               - avg_recall_experiment (float): the average number of items recalled per list 
    """

    num_participants = 0
    if words_recalled:
        # Determine the number of participants from the first list's data
        first_list_num = next(iter(words_recalled)) # first key in the dictionary 
        num_participants = len(words_recalled[first_list_num]) # length of inner list inside the first key 
    
    # Initialize participant-specific counters for lists with intrusions
    # Each list holds counts for a specific participant (indexed by participant_idx)
    # To store how many lists each participant had where a critical lure was intruded:
    participant_critical_lure_lists_count = [0] * num_participants
    # To store how many lists each participant had where a non-critical, non-studied word was intruded.
    participant_non_critical_lists_count = [0] * num_participants
    # To keep track of the total number of lists processed for each participant. The denominator for calculating probabilities per participant.
    participant_total_lists_count = [0] * num_participants # To count lists presented to each participant
    # To keep track of the total number of words recalled per list per participant 
    participant_recall_count = [0] * num_participants

    # Pre-process all critical lures for all participants to include expanded grammatical forms 
    expanded_critical_lures_perparticipant = []
    for p_idx in range(num_participants):
        participant_lures_original = critical_lures_perparticipant[p_idx]
        expanded_lures_for_p = set()
        for cl in participant_lures_original:
            expanded_lures_for_p.update(expand_word_forms(cl)) # Add original and all its forms
        expanded_critical_lures_perparticipant.append(expanded_lures_for_p)

    print(f'Items included in the critical lure recognition: {expanded_critical_lures_perparticipant}')
    # Iterate through each list number in the experiment
    # note: sorted ensures that the lists are in numerical order, not completely necessary 
    for list_number in sorted(words_recalled.keys()):
        # Iterate through each participant for the current list
        for participant_idx in range(num_participants):
            # participant_idx is the index of the inner list 
            recalled_words_set = set(words_recalled[list_number][participant_idx])
            presented_words_set = set(words_presented[list_number][participant_idx])

            # Number of recalled items per participant 
            participant_recall_count[participant_idx]+= len(recalled_words_set)

            # Get ALL critical lures for this specific participant (across all their lists)
            # This is the change: we take the full list of critical lures for the participant,
            # not just one specific to the current list_number.
            all_participant_critical_lures_set = set(expanded_critical_lures_perparticipant[participant_idx])

            # Identify words that were recalled but not presented (false recalls)
            false_recalls = recalled_words_set - presented_words_set

            # Increment the total count of lists processed for this participant
            participant_total_lists_count[participant_idx] += 1


            # Check for Critical Lure Intrusion on this list
            # An intrusion occurs if *any* of the participant's critical lures are found among the false recalls
            if len(false_recalls.intersection(all_participant_critical_lures_set)) > 0:
                participant_critical_lure_lists_count[participant_idx] += 1

            # Check for Non-Critical, Non-Studied Intrusion on this list
            # These are false recalls that are NOT found in the *entire set* of critical lures for this participant.
            non_critical_false_recalls = false_recalls - all_participant_critical_lures_set
            
            # An intrusion occurs if there is at least one such word
            if len(non_critical_false_recalls) > 0:
                participant_non_critical_lists_count[participant_idx] += 1


    # Calculate probabilities for each participant and then average across all participants
    all_participants_critical_lure_probs = []
    all_participants_non_critical_probs = []
    all_participants_recall_counts = [] 

    for participant_idx in range(num_participants):
        total_lists_for_participant = participant_total_lists_count[participant_idx]
        
        prob_critical_lure = 0.0
        prob_non_critical = 0.0
        avg_number_of_recalls = 0.0 
        # 1 list or more presented
        if total_lists_for_participant > 0:
            prob_critical_lure = participant_critical_lure_lists_count[participant_idx] / total_lists_for_participant
            prob_non_critical = participant_non_critical_lists_count[participant_idx] / total_lists_for_participant
            avg_number_of_recalls = participant_recall_count[participant_idx]/total_lists_for_participant
        all_participants_critical_lure_probs.append(prob_critical_lure)
        all_participants_non_critical_probs.append(prob_non_critical)
        all_participants_recall_counts.append(avg_number_of_recalls)

    # Calculate the final average probabilities across all participants
    avg_prob_critical_lure_intrusion = sum(all_participants_critical_lure_probs) / num_participants if num_participants > 0 else 0.0
    avg_prob_non_critical_intrusion = sum(all_participants_non_critical_probs) / num_participants if num_participants > 0 else 0.0
    avg_recall_experiment = sum(all_participants_recall_counts) / num_participants if num_participants > 0 else 0.0
    print(f'The average proportion of critical lure recall across {num_participants} participants studying {participant_total_lists_count[0]} lists: {avg_prob_critical_lure_intrusion}')
    print(f'The average proportion of non-critical lure intrusions across participants: {avg_prob_non_critical_intrusion}')
    print(f'The average number of words recalled across participants: {avg_recall_experiment}')
    return avg_prob_critical_lure_intrusion, avg_prob_non_critical_intrusion, avg_recall_experiment

calculate_recall(participant_recall, words_presented, critical_lures_perparticipant)

# Functions for Recognition results 


# Hit rate: the proportion of studied (or "old") words that a subject correctly identifies as "old"; original DRM (1995) Hit Rate .86 
# False alarm rate: the proportion of nonstudied (or "new," "lure") words that a subject incorrectly identifies as "old"
# In the (1995) study:
#       False alarm rate for critical lures: .84 
#       False alarm rate for weakly related lures: .21
#       False alarm rate for unrelated related lures: .02

#weak_lures_perparticipant

def calculate_recognition_metrics(
    recognition_tests,
    words_recognized,
    words_presented_in_study_phase, 
    critical_lures_perparticipant,
    weakly_related_lures_perparticipant, 
    verbose = False
):
    """
    Calculates hit rate and false alarm rates for critical lures, weakly related lures,
    and unrelated lures from recognition test results.

    Args:
        recognition_tests (list of list): Outer list index is participant_idx.
                                         Inner list contains ALL words presented to that participant
                                         in the recognition test across all their lists.
                                         Example: [['hatred', 'car', ...], [...]]
        words_recognized (list of list): Same structure as recognition_tests.
                                         Words that the participant correctly identified as "old".
        words_not_recognized (list of list): Same structure as recognition_tests.
                                            Words that the participant identified as "new".
        words_presented_in_study_phase (dict): From the study phase.
                                              Keys are list numbers (e.g., 1, 2) and values are
                                              lists of lists. Each inner list represents words studied
                                              by a specific participant for that list.
                                              Example: {1: [['apple', 'banana'], ['cat', 'dog']]}
        critical_lures_perparticipant (list of list): Outer list index is participant_idx.
                                                      Inner list contains ALL critical lures associated
                                                      with that participant (from their study lists).
                                                      Example: [['smoke', 'sleep'], ['chair', 'table']]
        weakly_related_lures_perparticipant (list of list): Same structure as critical_lures_perparticipant.
                                                            Contains ALL weakly related lures for that participant.

    Returns:
        tuple: A tuple containing:
               - avg_hit_rate (float)
               - avg_fa_rate_critical (float)
               - avg_fa_rate_weakly_related (float)
               - avg_fa_rate_unrelated (float)
    """

    num_participants = 0
    if recognition_tests:
        num_participants = len(recognition_tests)
        print(num_participants)
    # Initialize lists to store rates for each participant
    all_participants_hit_rates = []
    all_participants_fa_critical_rates = []
    all_participants_fa_weakly_related_rates = []
    all_participants_fa_unrelated_rates = []

    # Iterate through each participant
    for p_idx in range(num_participants):
        # Convert participant's lists to sets
        rec_test_words_for_p = set(recognition_tests[p_idx])
        recognized_words_for_p = set(words_recognized[p_idx])
        
        # Gather all words studied by this participant across ALL study lists
        studied_words_for_p = set()
        for list_num in words_presented_in_study_phase.keys():
            # Check if participant_idx is valid for this list_num's data
            if p_idx < len(words_presented_in_study_phase[list_num]):
                studied_words_for_p.update(words_presented_in_study_phase[list_num][p_idx])
        
        # Gather all critical and weakly related lures for this participant (from their study lists)
        critical_lures_for_p = set(critical_lures_perparticipant[p_idx])
        weakly_related_lures_for_p = set(weakly_related_lures_perparticipant[p_idx])

        # Print output of a singular participant for result verification purposes
        if p_idx == 0: # Results for a singular participant 
            print(f"\n--- DEBUGGING HIT RATE FOR PARTICIPANT {p_idx} ---")
            
            print(f"--- Recognition Test Details ---")
            print("Critical lures presented in recognition test:")
            print(critical_lures_for_p)
            print("Weakly Related luress presented in recognitin test:")
            print(weakly_related_lures_for_p)
            print("-" * 50)

            print(f"1. All words presented in recognition test (rec_test_words_for_p):")
            print(rec_test_words_for_p)
            print("-" * 50)

            print(f"2. All words studied by this participant across all lists (studied_words_for_p):")
            print(studied_words_for_p)
            print("-" * 50)

            # Calculate the intersection to find actual "old" words presented in the test
            old_words_in_rec_test = rec_test_words_for_p.intersection(studied_words_for_p)
            print(f"3. Intersection: Words that were studied AND presented in recognition test (old_words_in_rec_test):")
            print(old_words_in_rec_test)
            print(f"   Count of old_words_in_rec_test: {len(old_words_in_rec_test)}")
            print("-" * 50)

            print(f"4. Words this participant recognized as 'old' (recognized_words_for_p):")
            print(recognized_words_for_p)
            print("-" * 50)
            
            # Calculate the final hits
            hits_set = recognized_words_for_p.intersection(old_words_in_rec_test)
            hits_count = len(hits_set)
            print(f"5. Intersection: Words correctly recognized as old (hits_set):")
            print(hits_set)
            print(f"   Count of hits (hits_count): {hits_count}")
            print("-" * 50)

            total_old_words_presented = len(old_words_in_rec_test)
            print(f"6. Denominator for Hit Rate (total_old_words_presented): {total_old_words_presented}")
            
            hit_rate_individual = hits_count / total_old_words_presented if total_old_words_presented > 0 else 0.0
            print(f"7. Individual Hit Rate for Participant {p_idx}: {hit_rate_individual}")
            print(f"--- END DEBUGGING FOR PARTICIPANT {p_idx} ---\n")
        # --- END DEBUGGING PRINTS ---
        if verbose:
            print(f"\n--- RECOGNITION REPORT FOR PARTICIPANT {p_idx} ---") 

            print(f"--- Recognition Test Details ---")
            print("Critical lures presented in recognition test:")
            print(critical_lures_for_p)
            print("Weakly Related luress presented in recognitin test:")
            print(weakly_related_lures_for_p)
            print("-" * 50)

            print(f"1. All words presented in recognition test (rec_test_words_for_p):")
            print(rec_test_words_for_p)
            print("-" * 50)

            print(f"2. All words studied by this participant across all lists (studied_words_for_p):")
            print(studied_words_for_p)
            print("-" * 50)

            # Calculate the intersection to find actual "old" words presented in the test
            old_words_in_rec_test = rec_test_words_for_p.intersection(studied_words_for_p)
            print(f"3. Intersection: Words that were studied AND presented in recognition test (old_words_in_rec_test):")
            print(old_words_in_rec_test)
            print(f"   Count of old_words_in_rec_test: {len(old_words_in_rec_test)}")
            print("-" * 50)

            print(f"4. Words this participant recognized as 'old' (recognized_words_for_p):")
            print(recognized_words_for_p)
            print("-" * 50)
            
            # Calculate the final hits
            hits_set = recognized_words_for_p.intersection(old_words_in_rec_test)
            hits_count = len(hits_set)
            print(f"5. Intersection: Words correctly recognized as old (hits_set):")
            print(hits_set)
            print(f"   Count of hits (hits_count): {hits_count}")
            print("-" * 50)

            total_old_words_presented = len(old_words_in_rec_test)
            print(f"6. Denominator for Hit Rate (total_old_words_presented): {total_old_words_presented}")
            
            hit_rate_individual = hits_count / total_old_words_presented if total_old_words_presented > 0 else 0.0
            print(f"7. Individual Hit Rate for Participant {p_idx}: {hit_rate_individual}")
            print(f"--- END OF RECOGNITION REPORT FOR PARTICIPANT {p_idx} ---\n")
        # --- Categorize words presented in the recognition test for this participant ---
        
        # 1. Old (Studied) Words actually presented in the recognition test
        old_words_in_rec_test = rec_test_words_for_p.intersection(studied_words_for_p)

        # 2. Lures (Non-studied words) actually presented in the recognition test
        all_lures_in_rec_test = rec_test_words_for_p - studied_words_for_p

        # 3. Categorize lures into Critical, Weakly Related, and Unrelated
        
        # Critical Lures presented in recognition test
        critical_lures_presented = all_lures_in_rec_test.intersection(critical_lures_for_p)
        
        # Weakly Related Lures presented (that are not Critical Lures)
        temp_lures_excluding_critical = all_lures_in_rec_test - critical_lures_presented
        weakly_related_lures_presented = temp_lures_excluding_critical.intersection(weakly_related_lures_for_p)

        # Unrelated Lures presented (that are not Critical or Weakly Related Lures)
        unrelated_lures_presented = temp_lures_excluding_critical - weakly_related_lures_presented


        # --- Calculate Rates for this Participant ---

        # Hit Rate: (Correctly recognized Old words) / (Total Old words presented)
        hits_count = len(recognized_words_for_p.intersection(old_words_in_rec_test))
        total_old_words_presented = len(old_words_in_rec_test)
        hit_rate = hits_count / total_old_words_presented if total_old_words_presented > 0 else 0.0
        all_participants_hit_rates.append(hit_rate)

        # False Alarm Rate (Critical Lures): (Recognized Critical Lures) / (Total Critical Lures presented)
        fa_critical_count = len(recognized_words_for_p.intersection(critical_lures_presented))
        total_critical_lures_presented = len(critical_lures_presented)
        fa_rate_critical = fa_critical_count / total_critical_lures_presented if total_critical_lures_presented > 0 else 0.0
        all_participants_fa_critical_rates.append(fa_rate_critical)

        # False Alarm Rate (Weakly Related Lures): (Recognized Weakly Related Lures) / (Total Weakly Related Lures presented)
        fa_weakly_related_count = len(recognized_words_for_p.intersection(weakly_related_lures_presented))
        total_weakly_related_lures_presented = len(weakly_related_lures_presented)
        fa_rate_weakly_related = fa_weakly_related_count / total_weakly_related_lures_presented if total_weakly_related_lures_presented > 0 else 0.0
        all_participants_fa_weakly_related_rates.append(fa_rate_weakly_related)

        # False Alarm Rate (Unrelated Lures): (Recognized Unrelated Lures) / (Total Unrelated Lures presented)
        fa_unrelated_count = len(recognized_words_for_p.intersection(unrelated_lures_presented))
        total_unrelated_lures_presented = len(unrelated_lures_presented)
        fa_rate_unrelated = fa_unrelated_count / total_unrelated_lures_presented if total_unrelated_lures_presented > 0 else 0.0
        all_participants_fa_unrelated_rates.append(fa_rate_unrelated)

    # --- Average rates across all participants ---
    avg_hit_rate = sum(all_participants_hit_rates) / num_participants if num_participants > 0 else 0.0
    avg_fa_rate_critical = sum(all_participants_fa_critical_rates) / num_participants if num_participants > 0 else 0.0
    avg_fa_rate_weakly_related = sum(all_participants_fa_weakly_related_rates) / num_participants if num_participants > 0 else 0.0
    avg_fa_rate_unrelated = sum(all_participants_fa_unrelated_rates) / num_participants if num_participants > 0 else 0.0
    print(f'The average hit rate across participants is {avg_hit_rate}')
    print(f'The average false alarm rate for critical lures across participants is {avg_fa_rate_critical}')
    print(f'The average false alarm rate for weakly related lures across participants is {avg_fa_rate_weakly_related}')
    print(f'The average false alarm rate for unrelated lures across participants is {avg_fa_rate_unrelated}')
    return avg_hit_rate, avg_fa_rate_critical, avg_fa_rate_weakly_related, avg_fa_rate_unrelated

calculate_recognition_metrics(recognition_tests, participant_recog, words_presented, critical_lures_perparticipant, weakly_related_lures_perparticipant)


# Calculate average activation per list position across the experiment 
def avg_activation(activation_dict):
    all_lists = []

    #for value in data_dictionary.values():
    for value in activation_dict.values():
        all_lists.extend(value)

    if not all_lists:
        return []
    
    num_lists = len(all_lists)
    if not all_lists[0]:
        return []
    num_elements = len(all_lists[0])
    averaged_list = [0.0] * num_elements

    for i in range(num_elements):
        total = 0
        for inner_list in all_lists:
            # sum all lists at index i 
            total += inner_list[i]
        averaged_list[i] = total / num_lists

    return averaged_list

avg_activation_experiment = avg_activation(participant_recall_activation)
print(f'Average activation per serial position over the experiment: {avg_activation_experiment}')
# plot the average activation  

plt.plot(avg_activation_experiment)
plt.tick_params(axis='x', which='major', labelsize=5)
plt.xlabel("List Order")
plt.ylabel("Activation Level")
plt.title(f"Activation of Recalled Items to Position in List\nAveraged over {str(number_of_lists)} studied lists for {str(number_of_simulations)} participants")
plt.savefig("participant_recall_activation.png")


# %%
