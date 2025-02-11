class Natus_Config():
    """Preprocessing parameters for data collected with Natus system
    """

    def __init__(self) -> None:
       self.raw_root_path = '/mnt/data1/whb/SACM_Data/raw/'
       self.processed_root_path = '/mnt/data1/whb/SACM_Data/processed/'

       self.paradigm = 'V_48'
       self.subject_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06']
       self.segment_list = [[[], [], [], []],
                            [[], [], [], []],
                            [[], [], [], []],
                            [[], [], [], []],
                            [[], [], [], []],
                            [[], [], [], []]]

       self.session_list = [4, 4, 4, 4, 4, 4]

       # full channels
       self.channel_list = {'S01':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6'], # 38
                            'S02':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                                   'G1', 'G2', 'G3', 'G4', 'G5', 'G6'], # 54
                            'S03':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                                   'G1', 'G2', 'G3', 'G4', 'G5', 'G6'], # 54
                            'S04':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6'], # 46
                            'S05':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                                   'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8',
                                   'A`1', 'A`2', 'A`3', 'A`4', 'A`5', 'A`6', 'A`7', 'A`8', 
                                   'B`1', 'B`2', 'B`3', 'B`4', 'B`5', 'B`6', 'B`7', 'B`8', 
                                   'C`1', 'C`2', 'C`3', 'C`4', 'C`5', 'C`6', 'C`7', 'C`8', 
                                   'D`1', 'D`2', 'D`3', 'D`4', 'D`5', 'D`6'], # 86
                            'S06':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                                   'G1', 'G2', 'G3', 'G4', 'G5', 'G6']} # 54

       # SMC channel
       # self.single_channel_list = {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
       #                             'S02':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
       #                             'S03':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
       #                             'S04':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
       #                             'S05':['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'],  # wo/SMC
       #                             'S06':['G1', 'G2', 'G3', 'G4', 'G5', 'G6']}

       # single channel
       self.single_channel_list = {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                   'S02':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                   'S03':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                   'S04':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                   'S05':['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'],
                                   'S06':['G1', 'G2', 'G3', 'G4', 'G5', 'G6']}

       # bad channel
       self.bad_channel_list = {'S01':[],
                                'S02':[],
                                'S03':[],
                                'S04':[],
                                'S05':[],
                                'S06':[]}

       self.env_sample_rate = 200
       self.wav_sample_rate = 16000
       self.f_low = 70
       self.f_high = 170
       self.probe_channel_num = 8

       self.channel_path = 'read_full' # read_full, read_single_A, read_single_A_bipolar
       self.vad_half_window = 0.25

       self.label_save = True
       self.wav_save = True
       self.seeg_save = True

       self.is_VAD = False

       self.bef_plot = False
       self.aft_plot = False
       self.env_plot = False
       self.wav_plot = False

       self.save_mat = False



class Neuracle_Config():
    """Preprocessing parameters for data collected with Neuracle system
    """

    def __init__(self) -> None:
       self.raw_root_path = '/mnt/data1/whb/SACM_Data/raw/'
       self.processed_root_path = '/mnt/data1/whb/SACM_Data/processed/'

       self.paradigm = 'V_48'
       self.subject_list = ['S07', 'S08']
       self.segment_list = [[[1, 2], [1, 2], [1, 2], [1, 2]],
                            [[1], [1], [1], [1]]]

       self.session_list = [4, 4, 4]

       self.channel_list = {'S07':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6'], # 46
                            'S08':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                                   'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
                                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                                   'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 
                                   'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 
                                   'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                                   'G1', 'G2', 'G3', 'G4', 'G5', 'G6']} # 54

       # SMC channel
       # self.single_channel_list = {'S07':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'], wo/SMC
       #                             'S08':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']}
       
       # single channel
       self.single_channel_list = {'S07':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                   'S08':['G1', 'G2', 'G3', 'G4', 'G5', 'G6']}

       # bad channel
       self.bad_channel_list = {'S07':[],
                                'S08':[]}

       self.env_sample_rate = 200
       self.wav_sample_rate = 16000
       self.f_low = 70
       self.f_high = 170
       self.probe_channel_num = 8
       self.channel_path = 'read_full'
       self.vad_half_window = 0.25

       self.label_save = True
       self.wav_save = True
       self.seeg_save = True

       self.is_VAD = False

       self.bef_plot = False
       self.aft_plot = False
       self.env_plot = False
       self.wav_plot = False

       self.save_mat = False