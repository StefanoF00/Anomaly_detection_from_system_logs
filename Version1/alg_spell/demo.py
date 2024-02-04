import os
import sys
import logging
import pandas as pd
from Spell import LogParser
import argparse

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def deeplog_df_transfer(df, event_id_map):
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[['datetime', 'EventId']]
    df.loc[:, 'EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')


if __name__ == '__main__':
    ##########
    # Parser #
    ##########
    input_dir = 'data_to_parse/OpenStack_logs/'
    output_dir = 'alg_spell/spell_results/OpenStack_results/'
    log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
    dataset_dir = 'data_parsed/OpenStack/'

    # Message type threshold (default: 0.5)
    parser = argparse.ArgumentParser()          
    parser.add_argument('-tau', default=0.5, type=float)
    args = parser.parse_args()
    tau = args.tau
    print('tau={}'.format(str(tau)))

    # Regular expression list for optional preprocessing 
    regex      = [
        #r"[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}", # UUID
        r"((\d+\.){3}\d+,?)+", 
        r"/.+?\s", 
        r"\d+"         
    ]  

    parser = LogParser(
        indir=input_dir,
        outdir=output_dir,
        log_format=log_format,
        tau=tau,
        rex=regex
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for log_name in ['openstack_abnormal.log', 'openstack_normal2.log', 'openstack_normal1.log']:
        parser.parse(log_name)

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}/openstack_normal1.log_structured.csv')
    df_normal = pd.read_csv(f'{output_dir}/openstack_normal2.log_structured.csv')
    df_abnormal = pd.read_csv(f'{output_dir}/openstack_abnormal.log_structured.csv')

    event_id_map = dict()
    for i, event_id in enumerate(df['EventId'].unique(), 1):
        event_id_map[event_id] = i

    logger.info(f'length of event_id_map: {len(event_id_map)}')

    #########
    # Train #
    #########
    deeplog_train = deeplog_df_transfer(df, event_id_map)
    deeplog_file_generator(dataset_dir+'train', deeplog_train)

    ###############
    # Test Normal #
    ###############
    deeplog_test_normal = deeplog_df_transfer(df_normal, event_id_map)
    deeplog_file_generator(dataset_dir+'test_normal', deeplog_test_normal)

    #################
    # Test Abnormal #
    #################
    deeplog_test_abnormal = deeplog_df_transfer(df_abnormal, event_id_map)
    deeplog_file_generator(dataset_dir+'test_abnormal', deeplog_test_abnormal)