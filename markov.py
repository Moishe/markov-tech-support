import argparse
import json
import nltk
import random

from collections import defaultdict

all_words = defaultdict(list)
states = defaultdict(list)

ignore = ['<', '>', '``', '“']

def process_states(words):
    prev_word = None
    for word in words:
        all_words[word[0].lower()].append(word)
        if word in ignore:
            continue

        if prev_word:
            states[prev_word].append(word)
        prev_word = word

def create_sentence(current, max_depth, ideal=False, depth=0):
    if depth > max_depth:
        return current

    if current[-1] not in states:
        return current
    
    if ideal:
        counts = defaultdict(int)
        for word in states[current[-1]]:
            counts[word] += 1
        next_word = list(reversed(sorted(counts.items(), key=lambda item: item[1])))[0][0]
    else:
        next_word = random.choice(states[current[-1]])

    if next_word[0] in ['.', '?', '!']:
        return current + [next_word]

    return create_sentence(current + [next_word], max_depth, ideal, depth+1)

def main():  
    f = open('corpus/input-large.txt')

    parser = argparse.ArgumentParser("Tech support")
    parser.add_argument('--startword', type=str, default=None)
    parser.add_argument('--ideal', type=bool, default=False)
    args = parser.parse_args()

    for l in f:
        words = nltk.word_tokenize(l)
        #process_states(words)

        tagged = nltk.pos_tag(words)
        process_states(tagged)

    # pick a random starting word
    if args.startword:
        l = args.startword.lower()
        if l not in all_words:
            print("Couldn't find %s in the corpus, sorry!" % args.startword)
            exit(1)
        starting_word = random.choice(all_words[args.startword.lower()])
    else:
        starting_word = random.choice(list(states.keys()))
    
    result = create_sentence([starting_word], 30, args.ideal)
    result = [x[0] for x in result]
    result[0] = result[0].capitalize()
    print()
    print(' '.join(result))
    print()

if __name__ == "__main__":
    main()