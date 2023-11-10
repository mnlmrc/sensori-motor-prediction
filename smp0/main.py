from smp0.util import merge_blocks_mov, sort_by_probability

experiment = 'smp0'
participant_id = '100'
# block = '01'


probCues_index, tAx = sort_by_probability(experiment, participant_id, 91999)
