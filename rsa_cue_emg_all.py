import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--experiment', default='smp0', help='')
    parser.add_argument('--participants', default=['subj100',
                                                   'subj101',
                                                   'subj102',
                                                   'subj103',
                                                   'subj104',
                                                   'subj105',
                                                   'subj106',
                                                   'subj107',
                                                   'subj108',
                                                   'subj109',
                                                   'subj110'], help='')
    parser.add_argument('--method', default='crossnobis')

    args = parser.parse_args()

    experiment = args.experiment
    participants = args.participants
    method = args.method

    for participant in participants:
        os.system(f'python3 rsa_cue_emg_subj.py --participant_id {participant} --method {method}')
        # print(result.stdout.decode())
