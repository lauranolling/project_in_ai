from datamuse import datamuse
from transformers import pipeline
import syllables
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import re

# Models used to make rhymes
DEFAULT_SEN = "It is a long and winding road,"                  
API = datamuse.Datamuse()                                                               
# Tokenizer used to generate text
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")                                       
# Model To generate text
MODEL = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=TOKENIZER.eos_token_id)  # Model to generate text
UNMASKER = pipeline('fill-mask',model="bert-base-uncased",
    tokenizer="bert-base-uncased")                            

# Set seed to reproduce results.
tf.random.set_seed(10)                                         
#TOTAL_LINES = 10

def generate_song_structure():
    """
    Possibilities: 
    -   Number of different blocks 2,3,4,5              - song style
    -   Number of total blocks 3,4,5,6,7,8              - song style
    -   Number of lines in block 2,3,4,5,6,7,8          - block style

    -   4 lines pair rhyme AABB                     A     
    -   4 lines alternating rhyme sen ABAB          B    
    -   2 lines pair rhyme AA                       C  
    -   2 lines sign. sen + rhyme sen               E    
    -   1 line first word is sound word             G
    -   1 line random sentence                      H     
    -   1 line last word is sound word              M    
    -   1 line repeat previous sentence             R    
    -   Signature sentence                          S     
    -   Repeat word x3                              V 
    """

    song = []
    
    # Sample the number of blocks to include in the song
    blocks = int(random.sample(range(3,8),1)[0])
    repeats = int(random.sample(range(2,min((blocks),4)),1)[0])

    # Create block structures and add to song
    for _ in range(blocks - repeats + 1):
        block = ""
        lines = int(random.sample(range(2,8),1)[0])
        length = 0
        while length < lines:
            if (lines - length) >= 4:
                if len(block) == 0:
                    new = str(random.sample(['A','B','C','E'],1)[0])
                    if new in ['A','B']:
                        length = length + 4
                    else:
                        length = length + 2
                    block = block + new
                else:
                    new = str(random.sample(['A','B','C','E'],1)[0])
                    if new in ['A','B']:
                        length = length + 4
                    else:
                        length = length + 2
                    block = block + new

            elif (lines - len(block)) >= 2:
                if len(block) == 0:
                    new = str(random.sample(['C','E','G','H','M','S'],1)[0])
                    if new in ['C','E']:
                        length = length + 2
                    else:
                        length = length + 1
                    block = block + new 
                else:
                    new = str(random.sample(['C','E','G','H','M','R','S','V'],1)[0])
                    length = length + 1
                    block = block + new
                    
            else:
                block = block + str(random.sample(['G','H','M','R','S','V'],1)[0])
                length = length + 1
        song.append(block)
    
    # Add repeated blocks in song
    inds = random.sample(range(0,(blocks-1)),repeats)
    copies = []
    # If repeats only 2 only copy one block
    if repeats <= 3:
        copy = song[min(inds)]
        copies.append(copy)
        inds.remove(min(inds))
        while len(inds) >= 1:
            song.insert(min(inds),copy)
            inds.remove(min(inds))
    # Else, copy two blocks ad insert
    else:
        copy1 = song[min(inds)]
        copies.append(copy1)
        inds.remove(min(inds))
        copy2 = song[min(inds)]
        copies.append(copy2)
        inds.remove(min(inds))
        while len(inds) >= 1:
            if len(inds) % 2:
                song.insert(min(inds),copy1)
                inds.remove(min(inds))
            else:
                song.insert(min(inds),copy2)
                inds.remove(min(inds))


    """
    -   Repeat last sentence to end song L              - song style
    -   Start song with signature sentence  S           - song style
    -   Number of different blocks 2,3,4,5              - song style
    -   Number of total blocks 3,4,5,6,7,8              - song style
    -   End song with repeating last word x3 V          - song style
    """
    # Add song-stylistic features to song
    for i in range(2):
        style = str(random.sample(['R','S','V',''],1)[0])
        #print("style", style)
        if (i == 0) and (style == 'S'):
            song.insert(0,style)
        elif style != '':
            song.append(style)
        
    #print("song: ", song)
    return song, copies

def create_lyric(song, copies):
    """
    Function that creates lyrics given the song and block structures
    """
    lyric = []

    sentence = (input('Write a sentence to be the signature sentence of the song: ') or DEFAULT_SEN)
    #sentence = DEFAULT_SEN
    print("Signature sentence: ", sentence)

    print("song: ", song)

    
    lyrics = ""
    raw = generate_text(sentence)
    while len(raw) == 0:
        print("Couldn't generate text with the given sentence. Try something else.")
        sentence = (input('Write a sentence to be the signature sentence of the song: ') or DEFAULT_SEN)
        raw = generate_text(sentence)

    raw2 = []
    while len(raw2) == 0:
        new_sen = random.sample(raw,1)[0]
        raw2 = generate_text(new_sen)
    for line in raw2:
        raw.append(line)
    #print("raw: ", raw)

    sign_sen = raw.pop(0)

    finished = False
    copy = False
    copy_sens = dict()

    #---------------------------------------------------#
    # Edit raw text to match structure in "song" object #
    #---------------------------------------------------#

    while (len(raw)-4) and not(finished):        
        for block in song:
            if block in copies:
                copy = True
                if block in copy_sens:
                    sens = copy_sens[block]
                    for line in sens:
                        lyric.append(line)
                    copy = False
                    continue
                else:
                    sens = []
            #Make list of line styles
            #lyric.append('\n')
            lines = [l for l in block]
            # Check and create line in the desired style
            for line in lines:
                if line == 'A':
                    #print('A - pairwise rhymes')
                    # Check if a rhyme exists to line
                    found = False
                    first = ''
                    while not(found):
                        first = raw.pop(0)
                        first = mask_line(first.split())
                        word = first.split()[-1]
                        rhyme1 = find_rhyme(word)
                        if rhyme1 != 0:
                            found = True
                    # If rhyme exists, add line to lyric
                    lyric.append(first)
                    if copy == True:
                        sens.append(first)
                    #Create sentence to rhyme
                    second = raw.pop(0).split()
                    second.pop(-1)
                    second.append(rhyme1)
                    second = mask_line(second)
                    lyric.append(second)
                    if copy == True:
                        sens.append(second)

                    # Generate second rhyme pair
                    found = False
                    third = ''
                    while not(found):
                        third = raw.pop(0)
                        third = mask_line(third.split())
                        word = third.split()[-1]
                        rhyme2 = find_rhyme(word) 
                        if rhyme2 != 0:
                            found = True
                    lyric.append(third)
                    if copy == True:
                        sens.append(third)

                    fourth = raw.pop(0).split()
                    fourth.pop(-1)
                    fourth.append(rhyme2)
                    fourth = mask_line(fourth)
                    lyric.append(fourth)
                    if copy == True:
                        sens.append(fourth)
                
                elif line == 'B':
                    #print('B - alternating rhymes')
                    # Generate two first sentences
                    found = False
                    first = ''
                    while not(found):
                        first = raw.pop(0)
                        first = mask_line(first.split())
                        word1 = first.split()[-1]
                        rhyme1 = find_rhyme(word1)
                        if rhyme1 != 0:
                            found = True 
                    lyric.append(first)
                    if copy == True:
                        sens.append(first)

                    found = False
                    second = ''
                    while not(found):
                        second = raw.pop(0)
                        second = mask_line(second.split())
                        word2 = second.split()[-1]
                        rhyme2 = find_rhyme(word2)
                        if rhyme2 != 0:
                            found = True
                    lyric.append(second)
                    if copy == True:
                        sens.append(second)

                    # Generate rhyme sen.'s
                    third = (raw.pop(0)).split()
                    third.pop(-1)
                    third.append(rhyme1)
                    third = mask_line(third)
                    lyric.append(third)
                    if copy == True:
                        sens.append(third)

                    fourth = raw.pop(0).split()
                    fourth.pop(-1)
                    fourth.append(rhyme2)
                    fourth = mask_line(fourth)
                    lyric.append(fourth)
                    if copy == True:
                        sens.append(fourth)
                
                elif line == 'C':
                    #print('C - pairwise rhyme')
                    # Check if a rhyme exists to line
                    found = False
                    first = ''
                    while not(found):
                        first = raw.pop(0)
                        first = mask_line(first.split())
                        word = first.split()[-1]
                        rhyme1 = find_rhyme(word)
                        if rhyme1 != 0:
                            found = True
                    # If rhyme exists, add line to lyric
                    lyric.append(first)
                    if copy == True:
                        sens.append(first)

                    #Create sentence to rhyme
                    second = raw.pop(0).split()
                    second.pop(-1)
                    second.append(rhyme1)
                    second = mask_line(second)
                    lyric.append(second)
                    if copy == True:
                        sens.append(second)

                elif line == 'E':
                    #print('E - sign. sen + rhyme sen')
                    lyric.append(sign_sen)
                    if copy == True:
                        sens.append(sign_sen)
                    second = sign_sen.split()
                    word = second.pop(-1)
                    rhyme = find_rhyme(word)
                    if rhyme == 0:
                        rhyme = word

                    second.append(rhyme)
                    second = mask_line(second)
                    lyric.append(second)
                    if copy == True:
                        sens.append(second)

                elif line == 'G':
                    #print('G - first word is a sound word')
                    sen = raw.pop(0)
                    sen = mask_line(sen.split())
                    sen = sen.split()
                    word = sample_sound_word()
                    sen.insert(0,word)
                    sen = ' '.join(sen)
                    lyric.append(sen)
                    if copy == True:
                        sens.append(sen)

                elif line == 'H':
                    #print('H - sampled sentence')
                    sen = raw.pop(0)
                    sen = mask_line(sen.split())
                    lyric.append(sen)
                    if copy == True:
                        sens.append(sen)

                elif line == 'M':
                    #print('M - last word in sen is sound word')
                    sen = raw.pop(0)
                    sen = mask_line(sen.split())
                    sen = sen.split()
                    word = sample_sound_word()
                    sen.append(word)
                    sen = ' '.join(sen)
                    lyric.append(sen)
                    if copy == True:
                        sens.append(sen)

                elif line == 'R':
                    #print('R - previous sentence is repeated')
                    sen = str(lyric[-1])   
                    lyric.append(sen)
                    if copy == True:
                        sens.append(sen)

                elif line == 'S':
                    #print('S - insert signature sentence')
                    lyric.append(sign_sen)
                    if copy == True:
                        sens.append(sign_sen)

                elif line == 'V':
                    #print('V - repeat last word last sen x 3')
                    prev = lyric[-1].split()
                    word = prev.pop(-1)
                    sen = [word, word, word]
                    sen = ' '.join(sen)
                    lyric.append(sen)
                    if copy == True:
                        sens.append(sen)
                if copy == True:
                    copy_sens[block] = sens
        finished = True

    if not finished:
        "Ran out of text. Song couldn't finish"

    lengths = []
    for block in song:
        length = 0
        for letter in block:
            if letter in ['A','B']:
                length = length + 4
            elif letter in ['C','E']:
                length = length + 2
            else:
                length = length + 1
        lengths.append(length)

    # Make song in string format 
    for length in lengths:
        for line in range(length):
            lyrics = lyrics + lyric.pop(0) + "\n"
        lyrics = lyrics + "\n"
    
    song_text = open('lyrics.txt', 'w')
    song_text.writelines(lyrics)
    song_text.close()
    print("Saved lyrics in lyrics.txt!\n")
    
    return lyric

def find_rhyme(word):
    #print('Finding rhymes to the word: ', word)
    rhymes = API.words(rel_rhy=word, max=3)
    rhymes = {x['word'] for x in rhymes}

    list(rhymes)
    if len(rhymes) == 0:
        print('No rhymes found! trying again...')
        return 0


    rhyme = str(random.sample(rhymes,1))
    #clean up word
    rhyme = rhyme.replace('[\'','').replace('\']','') 

    return rhyme

def find_last(list_of_words):
    return list_of_words[-1]

def generate_text(sentence):
    """
    Created by following https://huggingface.co/blog/how-to-generate
    """
    print("Generating text...")
    input_text = TOKENIZER.encode(sentence, return_tensors='tf')

    sample_outputs = MODEL.generate(
    input_text,
    do_sample=True, 
    max_length=400, 
    top_k=100,
    top_p=0.95,
    num_return_sequences=1
    )

    new_lines = []
    for i, sample_output in enumerate(sample_outputs):
        new_lines.append(str(TOKENIZER.decode(sample_output, skip_special_tokens=True)))

    lyrics = []
    head, tail = 'None', 'None'
    
    for lyric in new_lines:
        lyric = lyric + '.'
        lyric = lyric.replace(', ','.').replace('. ', '.').replace('\n','').replace('["','').replace('"]','').replace('(','').replace(')','').replace('"','')

        while tail != '':
            head, _, tail = lyric.partition('.')

            # If sentence is longer than 11 words, split it into two 
            if len(head.split()) > 11:
                #https://stackoverflow.com/questions/3861674/split-string-by-number-of-words-with-python

                pieces = head.split()
                new_sens = (" ".join(pieces[i:i+11]) for i in range(0, len(pieces), 11))

                for i in new_sens:
                    lyrics.append(i)
            else:
                if len(head.split()) > 2:
                    lyrics.append(head)

            lyric = lyric.replace('.','',1)
            lyric = lyric.replace(head, '')
    for i, line in enumerate(lyrics):
        if len(line) <= 1:
            lyrics.pop(i)
            i = i + 1
    print("...Finished!")
    return lyrics

def mask_line(sentence):
    # Insert mask at random positions and unmask 10 times
    unmasked = ''
    if len(sentence) <= 2:
        return ' '.join(sentence)
    #print("sentence to mask: ", sentence)
    
    # Do not mask last word
    index_list = list(range(0, len(sentence)-2))
    for _ in range(15):
        i = random.sample(index_list,1)
        i = int(i[0])
        sentence[i] = '[MASK]'
        masked_line = ' '.join(sentence)
        word = find_word(masked_line)
        unmasked = str(masked_line.replace('[MASK]', word))
        sentence = unmasked.split()

    return unmasked

def find_word(sentence):
    #unmask sentence
    result = UNMASKER(sentence)

    suggestions = []
    {suggestions.append(result[x]['token_str']) for x in range(len(result))}   
    suggestions = {x.replace('Ġ', '') for x in suggestions}

    #for some reason changed from list to set
    suggestions = list(suggestions)
    word = str(random.sample(suggestions, 1))
    word = word.replace('[\'','').replace('\']','')
    return word

def sample_sound_word():
    return random.sample(['oh','woah','yeah','hey','ah'],1)[0]


if __name__ == "__main__":
    
    song, copies = generate_song_structure()
    lyrics = create_lyric(song, copies)
   

    

