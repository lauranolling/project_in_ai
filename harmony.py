import mido
import numpy as np
from deap import base, creator, tools
from mido import MidiFile, MidiTrack, Message, MetaMessage
import random
from os import listdir, walk
from os.path import isfile, join
import matplotlib.pyplot as plt

# For testing purposes
seed = random.seed(10)

def load_melodies():
    """
    Loads all midis from the 'melodies' folder into a list
    """
    filenames = []
    midis = []
    path = './melodies/'

    for (_,_,filename) in walk(path):
        filenames.extend(filename)
        break

    for midi in filenames:
        midis.append(MidiFile(str(path + midi),clip=True))

    return midis


def sample_melody(midis, len=4):
    """
    Creates melodies by sampling 4 random from the midis list, and starts and ends on a C.
    """
    melody = []
    notes = []
    
    # Set right type - 0 for not playing tracks at the same time, 
    # ticks_per_beat for the right tempo
    new_midi = MidiFile(ticks_per_beat=128,type=0)
    track = MidiTrack()

    # Random sample 4 melody sequences to make a whole sequence (16 4/4 
    # sequences)
    melody.append(random.sample(midis,len))
    # Melody is list of list of lists. Remove outer list
    melody = melody[0]

    # Append the two meta messages to beginning. For info only
    track.append(melody[0].tracks[0][0])
    track.append(melody[0].tracks[0][1])    

    # Replace first note with a C4 or C5
    start = melody[0].tracks[0][2].dict()['note']
    span = melody[0].tracks[0][3].dict()['time']
    melody[0].tracks[0].pop(2)
    melody[0].tracks[0].pop(2)

    # If bigger difference from C4 than C5
    if abs(start - 60) > abs(start - 72):
        # Insert C5
        new = mido.Message('note_on', note=72)
        new2 = mido.Message('note_off',note=72, time=span)
        melody[0].tracks[0].insert(2,new)
        melody[0].tracks[0].insert(3,new2)
    else:
        # Insert C4
        new = mido.Message('note_on', note=60)
        new2 = mido.Message('note_off',note=60, time=span)
        melody[0].tracks[0].insert(2,new)
        melody[0].tracks[0].insert(3,new2)
    
    # Concatenate the 4 separate midis into one
    for midi in melody:
        for msg in midi.tracks[0]:
            info = msg.bytes()
            # Do not add meta messages from beginning of the midis to concatenate
            if not msg.is_meta:
                track.append(msg)
                # Get the note value for each note played in the sequence
                if info[0] == 144:
                    notes.append(info[1])

    for msg in track:
        print(msg)
    print("\n\n")
    
    # Replace last note with a C4 or C5
    end = track[-2].dict()['note']
    span = track[-1].dict()['time']
    track.pop(-1)
    track.pop(-1)
    notes.pop(-1)

    # If bigger difference from C4 than C5
    if abs(end - 60) > abs(end - 72):
        # Insert C5
        new = mido.Message('note_on', note=72)
        new2 = mido.Message('note_off',note=72, time=span)
        track.append(new)
        track.append(new2)
        notes.append(72)
    else:
        # Insert C4
        new = mido.Message('note_on', note=60)
        new2 = mido.Message('note_off',note=60, time=span)
        track.append(new)
        track.append(new2)
        notes.append(60)

    for msg in track:
        print(msg)

    # Append end_of_file meta message
    track.append(melody[0].tracks[0][-1])
    # Add the concatenated track to a new midifile
    new_midi.tracks.append(track)

    new_midi.save('new_sample.mid')

    print("notes: ", notes)
    
    return new_midi, notes



def generate_harmonized_midi(best, notes):
    # Synchronous playing of tracks
    melody = MidiFile('new_sample.mid', clip=True)
    melody.type = 1
     # The three voices to be mixed
    soprano_notes = best[0:int(len(best)/2)]
    bass_notes = best[int(len(best)/2):int(len(best))]

    soprano_track =MidiTrack()
    bass_track = MidiTrack()

    # Append the two meta messages to beginning of new tracks. For info only
    soprano_track.append(melody.tracks[0][0])
    soprano_track.append(melody.tracks[0][1])  
    bass_track.append(melody.tracks[0][0])
    bass_track.append(melody.tracks[0][1])

    # Add note messages to track
    for ind, msg in enumerate(melody.tracks[0]):
        if not msg.is_meta:
            i = int(ind/2-2+1)
            if msg.dict()['type'] == 'note_on':
                soprano_msg = mido.Message('note_on', note=soprano_notes[i])
                bass_msg = mido.Message('note_on', note=bass_notes[i])
                soprano_track.append(soprano_msg)
                bass_track.append(bass_msg)
            else:
                note_span = msg.dict()['time']
                soprano_msg = mido.Message('note_off', note=soprano_notes[i], time=note_span)
                bass_msg = mido.Message('note_off', note=bass_notes[i], time=note_span)
                soprano_track.append(soprano_msg)
                bass_track.append(bass_msg)

    # Append end_of_file meta message
    soprano_track.append(melody.tracks[0][-1])
    bass_track.append(melody.tracks[0][-1])

    melody.tracks.append(soprano_track)
    melody.tracks.append(bass_track)

    print("soprano:")
    for msg in soprano_track:
        print(msg)
    print("bass:")
    for msg in bass_track:
        print(msg)
    print("melody:")
    for msg in melody.tracks[0]:
        print(msg)

    melody.save('harmonized_sample.mid')
    print('Added harmonies to melody!')




def generate_individual(chromosome_length):
    """
    Soprano midi range (60-81)
    Bass midi range (40-60)
    """
    # Create chromosome as a list with randomly sampled numbers in 
    # the range of the two voices: [soprano_notes,bass_notes]
    chromosome = []

    # Soprano notes
    chromosome.append(random.sample(range(60,81),int(chromosome_length/2)))
    # Bass notes
    chromosome.append(random.sample(range(40,60),int(chromosome_length/2)))
    
    # Flatten list
    chromosome = [item for sublist in chromosome for item in sublist]

    return chromosome



def evaluate(individual, weights, notes):
    """
    Write something about the function here
    Use penalties
    """
    # Necessary as it is nested in a list by the wrapper function
    individual = individual[0]

    
    # Retrieve notes for each of the harmony voices
    soprano = individual[0:int(len(individual)/2)]
    bass = individual[int(len(individual)/2):int(len(individual))]

    score = 0

    # Define rule parameters for evaluation (The same as the number of weights
    # defined)    
    mixing = 0                              #negative  
    parallel_fifths = 0                     #negative
    triad = 0                               #positive
    
    chromosome_length = len(individual)
    
    #-------------------------------------------------------------#
    # Determine the score from mixing (no mixing => better score) #
    #-------------------------------------------------------------#
    for note, note_soprano, note_bass in zip(notes, soprano, bass):
        if ((note_soprano - note) >= 0) and ((note - note_bass) >= 0):
            mixing = mixing + 1
        else:
            #print("voices mix")
            mixing = mixing - 1

    
    if mixing != 0:
        mixing = mixing/(chromosome_length/2)

    #--------------------------------------------------------------------#
    # Determine the score from parallel fifths (if none => better score) #
    #--------------------------------------------------------------------#
    j = 0
    prev_soprano = 0
    prev_bass = 0
    prev_note = 0
    
    for note, note_soprano, note_bass in zip(notes, soprano, bass):
        j = j + 1
        s1 = abs(prev_soprano - prev_note)
        s2 = abs(note_soprano - note)
        b1 = abs(prev_note - prev_bass)
        b2 = abs(note - note_bass)
        if j > 1:
            if not(((s1 == 7 and s2 == 7)) and not((b1 == 7 and b2 == 7))):
                parallel_fifths = parallel_fifths + 1
            else: 
                #print("perfect fifth parallel")
                parallel_fifths = parallel_fifths - 1
        prev_soprano = note_soprano 
        prev_bass = note_bass
        prev_note = note

    
    if parallel_fifths != 0:
        parallel_fifths = parallel_fifths/(chromosome_length-1)
    
    #------------------------------------------------------------------------#
    # Determine if there are any triads in major, minor or diminished chords #
    #------------------------------------------------------------------------#
    
    for note, note_soprano, note_bass in zip(notes, soprano, bass):
        s = note_soprano - note
        b = note - note_bass
        """
        if (s in [4,16]) and (b in [5,17]): # melody is the baseline
            #print("major chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        #else:
        #    triad = triad - 0.33
        
        if (s in [3,15]) and (b in [5,17]):
            #print("minor chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        #else:
        #    triad = triad - 0.33
        
        if (s in [3,15]) and (b in [6,18]):
            #print("diminished chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        if (s in [5,17]) and (b in [3,15]): # soprano is the baseline
            #print("major chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        if (s in [5,17]) and (b in [4,16]):
            #print("minor chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        if (s in [6,18]) and (b in [3,15]):
            #print("diminished chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        """
        # The bass harmony is the baseline
        if (s in [3,15]) and (b in [4,16]): 
            #print("major chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        if (s in [4,16]) and (b in [3,15]):
            #print("minor chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        
        if (s in [3,15]) and (b in [3,15]):
            #print("diminished chord! ", note_soprano, note, note_bass)
            triad = triad + 1
        
        
    
    #To avoid the score to grow out of proportion
    if triad != 0:
        triad = triad/(chromosome_length)


    #-----------------------------------------------------------------------#
    # Changing affecting criteria in scoring function. For testing purposes #
    #-----------------------------------------------------------------------#
    #mixing = 0                                  
    #parallel_fifths = 0                         #Rarely happens
    #triad = 0
    
    # Putting scoring criteria in a list
    criteria = [mixing, parallel_fifths, triad]

    # Add weighted scores to total score
    for w,c in zip(weights, criteria):
        #print("weight: ", w, "indiv: ", c)
        score = score + w*c

    return (score,)
    


def mutate(mutant,indpb):
    mutated = False 
    all_out = False
    while (mutated == False) and (all_out == False):
        for ind in range(len(mutant)):
            if random.random() < indpb:
                if ind < int(len(mutant)/2):
                    mutant[ind] = random.sample(range(60,81),1)[0]
                    mutated = True
                else:
                    mutant[ind] = random.sample(range(40,60),1)[0]
                    mutated = True
        all_out = True
    return mutant



def run_genetic(notes, weights,number_of_individuals, number_of_generations,individual_pb, tourn_size, cx_pb, mut_pb, lr):
    """
    Written using https://deap.readthedocs.io/en/master/examples/ga_onemax.html 
              and https://jooskorstanje.com/genetic-algorithm-optimize-your-diet.html
    """

    #print("notes: ", notes)
    chromosome_length = (len(notes)*2)

    # Used to make figures
    means = []
    stdevs = []
    gens = []

    #----------------------#
    # Set up the algorithm #
    #----------------------#

    # Fitness evaluation and individual type
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()

    toolbox.register("generate_individual", generate_individual,chromosome_length=chromosome_length)

    # n is how many times to call the function, here 1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.generate_individual, n=1) 
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, weights=weights, notes=notes)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=individual_pb)
    toolbox.register("select", tools.selTournament, tournsize=tourn_size)

    # Create population
    pop = toolbox.population(n=number_of_individuals)
    
    # Evaluate entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    #-------------------#
    # Perform evolution #
    #-------------------#
    
    # Extract all the fitnesses 
    fits = [ind.fitness.values[0] for ind in pop]

    # Number of generations
    g = 0

    while g < number_of_generations:
        g = g + 1
        cx_pb = cx_pb - lr
        mut_pb = mut_pb - lr
        individual_pb = individual_pb - lr
        #print("Generation %i\n" %g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation to the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Cross two individuals with probability cx_pb
            if random.random() < cx_pb:
                toolbox.mate(child1[0], child2[0])
                # Fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # Mutate an individual with probability mut_pb
            if random.random() < mut_pb:
                toolbox.mutate(mutant[0])
                #print("mutated! ", mutant)
                del mutant.fitness.values
        

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #print("  Evaluated %i individuals" % len(invalid_ind))
        # Replace the entire population with the offspring
        #print(offspring, "\n")
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        #print("  Min %s" % min(fits))
        #print("  Max %s" % max(fits))
        #print("  Avg %s" % mean)
        #print("  Std %s" % std)

        means.append(mean)
        stdevs.append(std)
        gens.append(g)

    
    print("-- End of evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    return best_ind, means, stdevs, gens

if __name__ == "__main__":

    midis = load_melodies()
    melody, notes = sample_melody(midis)
    
    num_inds = 200
    num_gens = 8500
    individual_pb = 0.015
    tourn_size = 70
    cx_pb = 0.8 
    mut_pb = 0.7
    lr = 0.00009
    weights = (3.0, 2.0, 10.0)
    best, means, stdevs, gens = run_genetic(notes=notes,
                       weights=weights,
                       number_of_individuals=num_inds, 
                       number_of_generations=num_gens,
                       individual_pb=individual_pb,
                       tourn_size=tourn_size,
                       cx_pb=cx_pb,
                       mut_pb=mut_pb,
                       lr=lr
                       )
    # Flatten list
    best = [item for sublist in best for item in sublist]
    #print ("best: ", best)
    sliding_window = [(sum(means[i:i+50])/50) for i in range(0, len(means), 50)]
    gens = [gens[i] for i in range(9,len(gens),50)]

    plt.figure()
    plt.plot(gens, sliding_window)
    plt.xlabel('Generations')
    plt.ylabel('Average Score')
    plt.savefig('learning_curve.pdf')
    plt.show()

    generate_harmonized_midi(best, notes)
    
    
