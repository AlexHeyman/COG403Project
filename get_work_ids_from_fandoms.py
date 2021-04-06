import sys
import subprocess
import urllib.parse


num_fics_to_retrieve = 5
python_command = 'py'


def get_work_ids_from_fandoms(input_path, output_path):
    input_file = open(input_path, 'r', encoding='utf8')
    # Input fandoms text file is divided into blocks of fandoms that should be
    # treated as related/redundant, separated by blank lines. For simplicity,
    # only consider the first fandom from each block, which we expect to be the
    # one with the most fics.
    last_line_was_space = True
    for line in input_file.readlines():
        line_is_space = line.isspace()
        if not line_is_space and last_line_was_space:
            # Strip away formatting stuff to just get the fandom name
            line = line[(line.index('. ') + 2):line.rindex(',')]
            print('Getting work ids from fandom: %s' % line)
            # Convert to URL encoding
            line = urllib.parse.quote_plus(line)
            # Get the desired AO3 search URL
            #url = r'https://archiveofourown.org/works?utf8=%E2%9C%93&work_search%5Bsort_column%5D=hits&work_search%5Bother_tag_names%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=F&work_search%5Bcomplete%5D=&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=en&commit=Sort+and+Filter&tag_id=' + line
            url = r'https://archiveofourown.org/works?utf8=%E2%9C%93&work_search%5Bsort_column%5D=hits&work_search%5Bother_tag_names%5D=&exclude_work_search%5Brating_ids%5D%5B%5D=13&exclude_work_search%5Brating_ids%5D%5B%5D=12&exclude_work_search%5Brating_ids%5D%5B%5D=9&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=F&work_search%5Bcomplete%5D=&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=en&commit=Sort+and+Filter&tag_id=' + line
            # Run ao3_work_ids.py
            subprocess.run([python_command, 'AO3Scraper/ao3_work_ids.py', url,
                            '--num_to_retrieve', str(num_fics_to_retrieve),
                            '--out_csv', output_path])
        last_line_was_space = line_is_space
    input_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python get_work_ids_from_fandoms.py [path to input fandoms text file] [path to output work ids csv file]')
        exit()
    
    get_work_ids_from_fandoms(sys.argv[1], sys.argv[2])
