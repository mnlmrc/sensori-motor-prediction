import argparse

import globals as gl
from force import save_npz

if __name__ == "__main__":
    # WAIT_TRIAL, // 0
    # START_TRIAL, // 1
    # WAIT_TR, // 2
    # WAIT_PLAN, // 3
    # WAIT_EXEC, // 4
    # GIVE_FEEDBACK, // 5
    # WAIT_ITI, // 6
    # ACQUIRE_HRF, // 7
    # END_TRIAL, // 8
    # experiment = sys.argv[1]
    # participant_id = sys.argv[2]
    # session = sys.argv[3]

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--experiment', default='smp2')
    parser.add_argument('--session', default='pilot')
    parser.add_argument('--participant_id', default='subj104')

    args = parser.parse_args()

    experiment = args.experiment
    session = args.session
    participant_id = args.participant_id

    # for participant_id in participants:
    save_npz(experiment, session, participant_id)


