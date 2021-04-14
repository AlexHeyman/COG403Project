import sys
import os
import re
import spacy


nlp = spacy.load('en_core_web_trf')


def preprocess_fics(input_path, output_path):
    for root, dirs, filenames in os.walk(input_path):
        for i in range(len(filenames)):
            filename = filenames[i]
            if not filename.endswith('.txt'):
                print('Skipping %s (%d of %d)' % (filename, i + 1, len(filenames)))
                continue
            print('Preprocessing %s (%d of %d)' % (filename, i + 1, len(filenames)))
            file = open(os.path.join(root, filename), 'r', encoding='utf8')
            data = file.read()
            file.close()
            # Remove leading and trailing whitespace
            data = data.strip()
            # Replace all whitespace characters with spaces
            data = re.sub(r'\s{1,}', ' ', data)
            # Remove duplicate spaces
            data = re.sub(r' +', ' ', data)

            doc = nlp(data)
            preprocessed_sentences = []
            for sentence in doc.sents:
                preprocessed_tokens = []
                for token in sentence:
                    # Replace token with its lemma
                    token_str = token.lemma_
                    if token_str[0] == '-' and token.text[0] != '-':
                        # unless the lemma begins with a dash and the token
                        # doesn't, in which case we still use the original token
                        token_str = token.text
                    # Append part of speech
                    token_str += '/' + token.pos_
                    preprocessed_tokens.append(token_str)
                preprocessed_sentences.append(' '.join(preprocessed_tokens))
            preprocessed_data = ' '.join(preprocessed_sentences)
            output_file = open(os.path.join(output_path, filename),
                               'w', encoding='utf8')
            output_file.write(preprocessed_data)
            output_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python preprocess_fics.py [path to input folder] [path to output folder]')
        exit()

    preprocess_fics(sys.argv[1], sys.argv[2])
