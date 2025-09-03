import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import soundfile as sf
import joblib
import scipy
import math

from scipy import signal as scipysignal
from sklearn.preprocessing import robust_scale
from pyedflib import highlevel
from label import Label_Ref
from preprocess_config import Natus_Config, Neuracle_Config
from pyAudioAnalysis import audioSegmentation as aS

class PreProcessor():
    def __init__(self, block_num=10, trial_num=48, trial_len=1.6) -> None:
        """SEEG data preprocessing script, compatible with both Natus & Neuracle systems

        Parameters
        -----------
        block_num: 
            The number of blocks for each session
        trial_num: 
            The number of trials per block, corresponding to 48 words
        trial_len:
            The duration of each trial in seconds, corresponding to the duration of each word in the experiment
        """

        self.block_num = block_num
        self.trial_num = trial_num
        self.trial_len = trial_len


    def readLabel(self, label_path, label_save_path, save):
        """Load labeling information: word order, time markers, etc., and convert word attributes into labels for storage.
        Includes word labels(0-47), initial labels(0-23), tone labels(0,1,2,3), initials, and finals

        Parameters
        -----------
        label_path: 
            Path to the original labeling file
        label_save_path: 
            Path to save the converted labels
        save:
            Whether to save the converted labels
        """

        label_file = os.path.join(label_path, 'final_label.final')
        label_read = joblib.load(label_file)
        self.play_info = np.array(label_read['label']['play_info'])
        self.wordlist_blocks = np.array(label_read['label']['wordlist_blocks'])
        # print('wordlist_blocks\n', self.wordlist_blocks)

        self.wordlist = self.wordlist_blocks.reshape(-1)
        label_ref = Label_Ref().label

        label_word = np.array([label_ref[word]['word'] for word in self.wordlist])
        label_initial_class = np.array([label_ref[word]['initial_class'] for word in self.wordlist])
        label_tone = np.array([label_ref[word]['tone'] for word in self.wordlist])
        label_initial = np.array([label_ref[word]['initial'] for word in self.wordlist])
        label_final = np.array([label_ref[word]['final'] for word in self.wordlist])

        if save == True:
            if not os.path.exists(label_save_path):
                os.makedirs(label_save_path)
            trans_label_file = os.path.join(label_save_path, 'trans_label.npz')
            np.savez(trans_label_file, word=label_word, 
                                       initial_class=label_initial_class, 
                                       tone=label_tone, 
                                       initial=label_initial, 
                                       final=label_final)

        # Recording start and end times, considered as the start and end times of each block within the session
        self.audio_info = []
        for i in range(self.block_num):
            self.audio_info.append(label_read['label'][f'audio_info_{i+1}'])
        # print('self.audio_info[0]', self.audio_info[0])


    def readSEEG(self, 
                 system, 
                 seeg_path, 
                 seeg_mat_save_path, 
                 segment, 
                 selected_channels, 
                 bad_channels, 
                 plot, 
                 save_mat):
        """Load SEEG data, select channels based on the channel list.

        The channel list is determined based on the subject's specific electrode setup and by removing bad channels 
        identified through observation. MNE lacks subsecond precision, with values in microvolts (μV) being scaled 
        by 1/1e6 based on pyedflib results. 

        Pyedflib supports subsecond precision with values in μV, and the scaling factor for subsecond precision 
        conversion should be adjusted from 100 to 10. (Bug of Pyedflib, see https://github.com/holgern/pyedflib/issues/236)
        The data acquisition format for the Neuracle system is BDF, and the starting time is known 
        to be an integer second. Therefore, MNE is still used to read data collected from the Neuracle system.

        Parameters
        -----------
        system: 
            The acquisition system, either Natus or Neuracle
        seeg_path: 
            Path to the SEEG data file
        seeg_mat_save_path: 
            Path to save the MAT-formatted data for acoustic contamination detection
        segment:
            List of data segments for saving, specific to Neuracle system
        selected_channels:
            SEEG channels to be load
        bad_channels: 
            Channels to be excluded as bad channels
        plot: 
            Whether to plot the raw data waveforms
        save_mat:
            Whether to save the MAT-formatted data for acoustic contamination detection
        """

        if system == 'Natus':
            self.selected_channels = selected_channels
            self.bad_channels = bad_channels
            self.pick_channels = list(set(self.selected_channels).difference(set(self.bad_channels)))
            self.pick_channels.sort(key=self.selected_channels.index)

            # print('selected_channels\n', self.selected_channels)
            # print('bad_channels\n', self.bad_channels)
            print('pick_channels\n', self.pick_channels)

            data_path = os.path.join(seeg_path, 'data.edf')
            signals, signal_headers, header = highlevel.read_edf(data_path, ch_names=self.pick_channels, verbose=True)
            signals = signals * 1e-6

            # print('header\n', header)
            # print('header[startdate]\n', header['startdate'])
            print('sample_rate:', signal_headers[0]['sample_rate'])

            self.fs = signal_headers[0]['sample_rate']
            self.meas_date = header['startdate']
            self.SEEG = np.array(signals)


        elif system == 'Neuracle':
            self.SEEG = []
            for i in range(len(segment)):
                data_path = os.path.join(seeg_path, f'{segment[i]}/data.bdf')
                raw = mne.io.read_raw_bdf(data_path, preload=False)
                print('raw.info\n', raw.info)
    
                if i == 0:
                    self.fs = raw.info['sfreq']
                    self.ch_names = raw.info['ch_names']
                    self.meas_date = raw.info['meas_date']
    
                if plot == True:
                    raw.plot(duration=10, start=600, n_channels=len(self.ch_names))
                    raw.compute_psd(fmax=300).plot()
                    plt.show()
                
                picks = mne.pick_channels(self.ch_names, include=selected_channels, exclude=bad_channels)
                self.pick_channels = np.array(self.ch_names)[picks]
                print('self.pick_channels\n', self.pick_channels)
                raw = raw.pick_channels(self.pick_channels)
                data, times = raw[:, :]
                self.SEEG.append(data)
    
            self.SEEG = np.concatenate(self.SEEG, axis=1)
            

        # Clipped raw SEEG data start time
        self.meas_date = pd.to_datetime(self.meas_date.strftime("%Y-%m-%d %H:%M:%S.%f"))
        print('self.meas_date:', self.meas_date)

        # Convert to MAT format and save for acoustic contamination detection
        if save_mat == True:
            if not os.path.exists(seeg_mat_save_path):
                os.makedirs(seeg_mat_save_path)
            for i in range(self.block_num):
                start_time, end_time = self.audio_info[i][0]
                start_time = pd.to_datetime(start_time)
                end_time = pd.to_datetime(end_time)
                start = (start_time-self.meas_date).seconds + (start_time-self.meas_date).microseconds / 1e6
                end = (end_time-self.meas_date).seconds + (end_time-self.meas_date).microseconds / 1e6
                if i == 0:
                    seeg_mat = self.SEEG[:, int(start*self.fs):int(end*self.fs)]
                else:
                    seeg_mat = np.hstack((seeg_mat, self.SEEG[:, int(start*self.fs):int(end*self.fs)]))

            # Ensure uniform length
            seeg_mat = seeg_mat[:, 0:773*2048]
            print('seeg_mat.shape:', seeg_mat.shape)
            scipy.io.savemat(os.path.join(seeg_mat_save_path, 'seeg_mat.mat'), {'data':seeg_mat.T})

        # The start time of the first recording and the end time of the last recording, used to determine the actual experiment duration
        print('self.audio_info[0]:', self.audio_info[0])
        print('self.audio_info[-1]:', self.audio_info[-1])
        self.exp_start_time, _ = self.audio_info[0][0]
        _, self.exp_end_time = self.audio_info[-1][0]
        self.exp_start_time = pd.to_datetime(self.exp_start_time)
        self.exp_end_time = pd.to_datetime(self.exp_end_time)
        print('self.exp_start_time:', self.exp_start_time)
        print('self.exp_end_time:', self.exp_end_time)

        # Use meas_date as the anchor point to calculate the trimming points (samples not seconds) based on start_time and end_time
        diff_start = (self.exp_start_time-self.meas_date).seconds + (self.exp_start_time-self.meas_date).microseconds / 1e6
        diff_end = (self.exp_end_time-self.meas_date).seconds + (self.exp_end_time-self.meas_date).microseconds / 1e6
        self.SEEG = self.SEEG[:, int(diff_start*self.fs):int(diff_end*self.fs)]

        print('SEEG.shape:', self.SEEG.shape)
        # print('SEEG\n', self.SEEG)


    def dataDetrend(self):
        """linear detrend
        """

        self.SEEG = scipysignal.detrend(self.SEEG, axis=-1)


    def reReference(self):
        """CAR (Common Average Referencing): Subtract the average of all selected channels for re-referencing
        """

        Ref_data = np.mean(self.SEEG, axis=0)
        self.SEEG = self.SEEG - Ref_data


    def reReference_Bipolar(self, probe_channel_num):
        """Bipolar re-referencing: Calculate the difference between adjacent electrodes

        Parameters
        -----------
        probe_channel_num: 
            The number of contacts on each electrode, used to handle cases where physically non-adjacent electrodes are numbered consecutively
        """
        
        channel_num = self.SEEG.shape[0]
        probe_num = math.ceil(channel_num / probe_channel_num)

        # Mono-electrod: No need to handle the boundary cases of physically non-adjacent electrodes with consecutive numbers
        if channel_num <= probe_channel_num:
            self.SEEG = np.diff(self.SEEG, axis=0)

        # Multi-electrode: Need to remove bipolar channels with consecutive electrode numbers that are physically non-adjacent, e.g., "A8-B1"
        else:
            self.SEEG = np.diff(self.SEEG, axis=0)
            edge_index = [probe_channel_num-1 + i*probe_channel_num for i in range(probe_num-1) if probe_channel_num-1 + i*probe_channel_num < channel_num]
            mask = np.ones(self.SEEG.shape[0], dtype=bool)
            mask[edge_index] = False
            self.SEEG = self.SEEG[mask]
        

    def filter(self, f_low=70, f_high=170):
        """SEEG bandpass filtering, along with 50Hz and its harmonics notch filtering

        Parameters
        -----------
        f_low: 
            The lower cutoff frequency for the bandpass filter
        f_high: 
            The upper cutoff frequency for the bandpass filter
        """

        b1, a1 = scipysignal.iirnotch(w0=50, Q=30, fs=self.fs)
        self.SEEG = scipysignal.filtfilt(b1, a1, self.SEEG)
        b2, a2 = scipysignal.iirnotch(w0=100, Q=30, fs=self.fs)
        self.SEEG = scipysignal.filtfilt(b2, a2, self.SEEG)
        b3, a3 = scipysignal.iirnotch(w0=150, Q=30, fs=self.fs)
        self.SEEG = scipysignal.filtfilt(b3, a3, self.SEEG)

        b_band, a_band = scipysignal.butter(N=4, Wn=(f_low, f_high), btype='bandpass', fs=self.fs)

        # filtfilt zero-phase filtering (forward and reverse), with filter order of 2*N
        self.SEEG = scipysignal.filtfilt(b_band, a_band, self.SEEG)


    def clamping(self, plot):
        """Robust scaling & clamping

        Parameters
        -----------
        plot: 
            Whether to plot the waveform and its statistical data (e.g., energy in different frequency bands) after preprocessing but before envelope extraction
        """

        self.SEEG = robust_scale(self.SEEG, axis=1)
        self.SEEG = np.clip(self.SEEG, a_min=-15, a_max=15)

        if plot == True:
            info = mne.create_info(list(self.pick_channels), self.fs)
            raw_new = mne.io.RawArray(self.SEEG, info)
            raw_new.plot(duration=10, start=600, n_channels=len(self.pick_channels))


    def hilbertTrans(self, plot):
        """Calculate the envelope using the Hilbert transform

        Parameters
        -----------
        plot: 
            Whether to plot the envelope
        """

        self.envelope = np.abs(scipysignal.hilbert(self.SEEG))
        print('envelope.shape:', self.envelope.shape)

        if plot == True:
            t1 = np.arange(1000) / 1000
            t2 = np.arange(1000) / 1000
            t3 = np.arange(1000) / 1000
            fig, ax0 = plt.subplots(nrows=1)
            ax0.plot(t1, self.SEEG[0][:1000], label='seeg_data')
            ax0.plot(t2, self.envelope[0][:1000], label='envelope_data')
            ax0.plot(t3, np.abs(self.SEEG[0][:1000]), label='abs_seeg')
            ax0.set_xlabel("time in seconds")
            ax0.legend()
            fig.tight_layout()
    
    def dataResample(self, new_sample_rate):
        """Envelope downsampling

        Parameters
        -----------
        new_sample_rate: 
            The new sampling rate after downsampling
        """

        down_factor = int(self.fs / new_sample_rate)
        self.envelope = scipysignal.resample(self.envelope, self.envelope.shape[1] // down_factor, axis=1)
        print('Resample.shape:', self.envelope.shape)


    def dataSplit(self, 
                  seeg_save_path, 
                  new_sample_rate, 
                  save=False, 
                  is_VAD=False, 
                  vad_half_window=0.5):
        
        """SEEG data segmentation

        Parameters
        -----------
        seeg_save_path: 
            Path to save the segmented SEEG data
        new_sample_rate: 
            New sampling rate for the SEEG data
        save: 
            Whether to save the segmented data
        is_VAD: 
            Whether to segment SEEG based on VAD (Voice Activity Detection) results
        vad_half_window: 
            The half-window length for VAD detection
        """

        # Assign time labels to the segmented data
        envelope_index = pd.date_range(start=self.exp_start_time, end=self.exp_end_time, periods=self.envelope.shape[1])
        df_envelope = pd.DataFrame(data=self.envelope.T, index=envelope_index)

        # Segment trials based on play_info (time labels) and ensure uniform number of samples
        self.envelope_rest = []
        self.envelope_word = []

        for index, (s, e) in enumerate(self.play_info):
            seeg_trial = np.array(df_envelope[s:e])[:int(self.trial_len*new_sample_rate), :].T

            if is_VAD == True:
                word_segment_center = np.mean(self.word_segments[index], axis=-1)
                pad_trial = np.pad(seeg_trial, ((0, 0), (new_sample_rate, new_sample_rate)), 'wrap')
                window_start = int(new_sample_rate + (word_segment_center-vad_half_window)*new_sample_rate)
                window_end = window_start + int(2*vad_half_window*new_sample_rate)
                word = pad_trial[:, window_start:window_end]

                rest1 = pad_trial[:, int(1*new_sample_rate):window_start] # Resting state before VAD segment
                rest2 = pad_trial[:, window_end:int(2.6*new_sample_rate)] # Resting state after VAD segment

                # Non-speech segment was selected from the remaining part of the trial distant from the speech center
                if rest2.shape[-1] >= word.shape[-1]:
                    rest = rest2[:, -int(2*vad_half_window*new_sample_rate):]
                else:
                    rest = rest1[:, :int(2*vad_half_window*new_sample_rate)]

            else:
                word = seeg_trial[:, int(0*new_sample_rate):int(1.6*new_sample_rate)]
                rest = seeg_trial[:, int(0*new_sample_rate):int(0.5*new_sample_rate)]

            self.envelope_word.append(word)
            self.envelope_rest.append(rest)

        self.envelope_word = np.array(self.envelope_word)
        self.envelope_rest = np.array(self.envelope_rest)

        if save == True:
            if not os.path.exists(seeg_save_path):
                os.makedirs(seeg_save_path)
            np.save(os.path.join(seeg_save_path, 'envelope_word.npy'), self.envelope_word)
            np.save(os.path.join(seeg_save_path, 'envelope_rest.npy'), self.envelope_rest)


    def wavSplit(self, 
                 wav_path, 
                 wav_save_path, 
                 wav_mat_save_path, 
                 new_sample_rate, 
                 save=False, 
                 plot=False, 
                 is_VAD=False, 
                 save_mat=False, 
                 vad_half_window=0.25):
        
        """WAV data segmentation

        Parameters
        -----------
        wav_path: 
            Path to load the audio data
        wav_save_path: 
            Path to save the processed audio
        wav_mat_save_path: 
            Path to store the audio in MAT format for acoustic contamination detection
        new_sample_rate: 
            New sampling rate for the audio data
        save: 
            Whether to save the processed audio
        plot: 
            Whether to plot the audio waveform
        is_VAD: 
            Whether to perform VAD (Voice Activity Detection)
        save_mat: 
            Whether to save the audio in MAT format for acoustic contamination detection
        vad_half_window: 
            The half-window length for VAD detection
        """

        self.wav_rest = []
        self.wav_word = []
        self.word_segments = []

        for i in range(self.block_num):
            wav_file = os.path.join(wav_path, f'{i+1}.wav')
            old_wav, old_sample_rate = sf.read(wav_file)
            old_wav = old_wav.reshape(-1, old_wav.shape[0])

            # create mono-channel audio
            old_wav = np.mean(old_wav, axis=0)
            new_audio_signal = librosa.resample(old_wav, orig_sr=old_sample_rate, target_sr=new_sample_rate)

            # Convert to MAT format and save for acoustic contamination detection
            if save_mat == True:
                if not os.path.exists(wav_mat_save_path):
                    os.makedirs(wav_mat_save_path)
                wav_mat_signal = librosa.resample(old_wav, orig_sr=old_sample_rate, target_sr=2048)
                print('wav_mat_signal.shape:', wav_mat_signal.shape)
                if i == 0:
                    wav_mat = wav_mat_signal
                else:
                    wav_mat = np.hstack((wav_mat, wav_mat_signal))

            wav_start_time, wav_end_time = pd.to_datetime(self.audio_info[i][0])

            wav_index = pd.date_range(start=wav_start_time, end=wav_end_time, periods=new_audio_signal.shape[0])
            wav_Frame = pd.DataFrame(data=new_audio_signal.T, index=wav_index)

            # Segment the audio based on the number of trials in each block and ensure uniform length
            for index, (s, e) in enumerate(self.play_info[int(self.trial_num*i):int(self.trial_num*(i+1))]):
                wav_trial = np.array(wav_Frame[s:e])[:int(self.trial_len*new_sample_rate)].T

                if is_VAD == True:
                    word_segment = np.array(aS.silence_removal(wav_trial.T, new_sample_rate, 0.02, 0.02, 0.5, weight=0.6, plot=False))
                    if not word_segment.size == 2:
                        word_segment = self.segments_correct(word_segment, self.word_segments)
                    self.word_segments.append(word_segment)
                    word_segment_center = np.mean(word_segment, axis=-1)

                    pad_trial = np.pad(wav_trial, ((0, 0), (new_sample_rate, new_sample_rate)), 'wrap')
                    window_start = int(new_sample_rate + (word_segment_center-vad_half_window)*new_sample_rate)
                    window_end = window_start + int(2*vad_half_window*new_sample_rate)

                    word = pad_trial[:, window_start:window_end]
                    rest1 = pad_trial[:, int(1*new_sample_rate):window_start]
                    rest2 = pad_trial[:, window_end:int(2.6*new_sample_rate)]

                    # Non-speech segment was selected from the remaining part of the trial distant from the speech center
                    if rest2.shape[-1] >= word.shape[-1]:
                        rest = rest2[:, -int(2*vad_half_window*new_sample_rate):]
                    else:
                        rest = rest1[:, :int(2*vad_half_window*new_sample_rate)]

                    if plot == True:
                        t = np.arange(int(3.6*new_sample_rate)) / int(new_sample_rate)
                        fig, ax = plt.subplots(nrows=1)
                        plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
                        print('word:', self.wordlist[int(self.trial_num*i+index)])
                        ax.plot(t, pad_trial[0], label=self.wordlist[int(self.trial_num*i+index)])
                        ax.set_xlabel("time in seconds")
                        ax.axvline(x=window_start / int(new_sample_rate), color='b')
                        ax.axvline(x=window_end / int(new_sample_rate), color='b')
                        ax.legend(self.wordlist[int(self.trial_num*i+index)])
                        plt.show()

                else:
                    word = wav_trial[:, int(0*new_sample_rate):int(1.6*new_sample_rate)]
                    rest = wav_trial[:, int(0*new_sample_rate):int(0.5*new_sample_rate)]

                    # Plot the audio waveform
                    if plot == True:
                        t = np.arange(int(1.6*new_sample_rate)) / int(new_sample_rate)
                        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
                        plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
                        print('word:', self.wordlist[int(self.trial_num*i+index)])
                        ax1.plot(t, wav_trial[0], label=self.wordlist[int(self.trial_num*i+index)])
                        ax1.set_xlabel("time in seconds")
                        ax1.legend(self.wordlist[int(self.trial_num*i+index)])
                        plt.show()

                self.wav_word.append(word)
                self.wav_rest.append(rest)
        
        self.wav_word = np.array(self.wav_word)
        self.wav_rest = np.array(self.wav_rest)
        self.word_segments = np.array(self.word_segments)

        if save == True:
            if not os.path.exists(wav_save_path):
                os.makedirs(wav_save_path)
            
            np.save(os.path.join(wav_save_path, 'wav_word.npy'), self.wav_word)
            np.save(os.path.join(wav_save_path, 'wav_rest.npy'), self.wav_rest)
        
        if save_mat == True:
            wav_mat = wav_mat.reshape(1, -1)[:, 0:773*2048]
            print('wav_mat.shape:', wav_mat.shape)
            scipy.io.savemat(os.path.join(wav_mat_save_path, 'wav_mat.mat'), {'data':wav_mat.T})


    def segments_correct(self, word_segment, word_segments):
        """Combine "standard speech" segments detected as a single speech activity, and compute the average and time center

        Parameters
        -----------
        word_segment: 
            The current anomalous speech activity segment to be corrected, which refers to a segment where no speech activity was detected or more than one speech activity was detected
        word_segments: 
            The currently processed normal speech activity segments, serving as the correction reference library
        """

        non_empty_segments = [segments for segments in word_segments if segments.size == 2]
        non_empty_segments = np.array(non_empty_segments)
        # Vertical time center, i.e., the average start time and average end time of the speech activity
        if len(non_empty_segments) > 0:
            segments_vavg = np.mean(non_empty_segments, axis=0)
        else:
            segments_vavg = np.array([[0.4, 1.4]])
            
        # For segments where no speech is detected or multiple speech segments are detected, use the average start and end times from other speech segments
        word_segment = segments_vavg
        return word_segment


def sessionFolderFind(root_path, session_number):
    """Find the SEEG data folder containing the current session_number in the root_path directory

    Parameters
    -----------
    root_path: 
        The root directory
    session_number: 
        The session index for which the file location needs to be found
    """

    folder_names = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    session_str = str(session_number)

    for folder in folder_names:
        if session_str in folder.split('_'):
            seeg_path = os.path.join(root_path, folder)
            return seeg_path


def run_preprocess_word(system,
                        raw_root_path, 
                        processed_root_path, 
                        paradigm, 
                        subject_list, 
                        segment_list, 
                        session_list, 
                        channel_list, 
                        single_channel_list, 
                        bad_channel_list, 
                        env_sample_rate, 
                        wav_sample_rate, 
                        f_low, 
                        f_high,
                        probe_channel_num, 
                        channel_path, 
                        vad_half_window, 
                        label_save, 
                        wav_save, 
                        seeg_save, 
                        is_VAD, 
                        bef_plot, 
                        aft_plot, 
                        env_plot,
                        wav_plot, 
                        save_mat):
    
    """Normal data processing for tasks that do not involve VAD, such as 48-word classification, 4-tone classification, etc.

    Parameters
    -----------
    system:
        The data acquisition system, either Natus or Neuracle
    raw_root_path: 
        Path to load the raw dataset
    processed_root_path: 
        Path to store the processed dataset
    paradigm:
        Experimental paradigm version
    subject_list: 
        List of subjects
    segment_list:
        List of segments for data storage, specific to the Neuracle system
    session_list: 
        List of experimental sessions
    channel_list:
        List of channels
    single_channel_list: 
        List of single electrodes
    bad_channel_list:
        List of bad channels
    env_sample_rate:
        Sampling rate after envelope downsampling
    wav_sample_rate:
        Sampling rate after audio downsampling
    f_low: 
        Low-frequency cutoff for bandpass filtering
    f_high: 
        High-frequency cutoff for bandpass filtering
    probe_channel_num:
        Number of contacts on a single electrode when fully loaded
    channel_path:
        Data storage version: 'read_good_single' for single electrodes; 'read_good' for multi-electrode setup
    vad_half_window:
        Half-window length for voice activity detection (VAD)
    label_save:
        Whether to save labels
    wav_save:
        Whether to save processed audio data
    seeg_save:
        Whether to save processed SEEG data
    is_VAD:
        Whether to perform voice activity detection (VAD)
    bef_plot:
        Whether to plot data before processing (raw data plot)
    aft_plot:
        Whether to plot data after processing (processed data plot, before envelope extraction)
    env_plot:
        Whether to plot the envelope
    wav_plot:
        Whether to plot audio data
    save_mat:
        Whether to store data in .mat format for use with MATLAB's acoustic contamination check toolkit
    """
    
    for subject_id in range(len(subject_list)):
        for i in range(session_list[subject_id]):
            label_path = raw_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/label_read/session{i+1}'
            wav_path = raw_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/audio/session{i+1}'
            seeg_path = sessionFolderFind(raw_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/seeg', i+1)

            label_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/trans_label'  
            wav_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/wav/audio'
            wav_mat_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/{channel_path}/wav_mat'
            
            seeg_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/{channel_path}/seeg'
            seeg_mat_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/{channel_path}/seeg_mat'
            

            # label
            preprocessor = PreProcessor(block_num=10, trial_num=48, trial_len=1.6)
            preprocessor.readLabel(label_path, label_save_path, save=label_save)

            # wav
            preprocessor.wavSplit(wav_path, 
                                  wav_save_path, 
                                  wav_mat_save_path, 
                                  new_sample_rate=wav_sample_rate, 
                                  save=wav_save, 
                                  plot=wav_plot, 
                                  is_VAD=is_VAD, 
                                  save_mat=save_mat, 
                                  vad_half_window=vad_half_window)
            print('preprocessor.wav_cross.shape:', preprocessor.wav_rest.shape)
            print('preprocessor.wav_word.shape:', preprocessor.wav_word.shape)


            # seeg
            if 'single' in channel_path:
                preprocessor.readSEEG(system, 
                                      seeg_path, 
                                      seeg_mat_save_path, 
                                      segment_list[subject_id][i], 
                                      single_channel_list[subject_list[subject_id]], 
                                      bad_channel_list[subject_list[subject_id]], 
                                      plot=bef_plot, 
                                      save_mat=save_mat)
            else:
                preprocessor.readSEEG(system, 
                                      seeg_path, 
                                      seeg_mat_save_path, 
                                      segment_list[subject_id][i], 
                                      channel_list[subject_list[subject_id]], 
                                      bad_channel_list[subject_list[subject_id]], 
                                      plot=bef_plot, 
                                      save_mat=save_mat)

            preprocessor.dataDetrend()

            if 'bipolar' in channel_path:
                preprocessor.reReference_Bipolar(probe_channel_num)
                print("reReference_Bipolar apply")
            else:
                preprocessor.reReference()
                print("reReference apply")

            preprocessor.filter(f_low=f_low, f_high=f_high)
            preprocessor.clamping(plot=aft_plot)
            preprocessor.hilbertTrans(plot=env_plot)
            preprocessor.dataResample(env_sample_rate)

            preprocessor.dataSplit(seeg_save_path, 
                                   env_sample_rate, 
                                   save=seeg_save, 
                                   is_VAD=is_VAD, 
                                   vad_half_window=vad_half_window)
            print('preprocessor.envelope_word.shape:', preprocessor.envelope_word.shape)
            print('preprocessor.envelope_rest.shape:', preprocessor.envelope_rest.shape)

    if label_save == False:
        print("label not saved !!!")
    if wav_save == False:
        print("wav not saved !!!")
    if seeg_save == False:
        print("seeg not saved !!!")


def run_preprocess_detect(system,
                          raw_root_path, 
                          processed_root_path, 
                          paradigm, 
                          subject_list, 
                          segment_list, 
                          session_list, 
                          channel_list, 
                          single_channel_list, 
                          bad_channel_list, 
                          env_sample_rate, 
                          wav_sample_rate, 
                          f_low, 
                          f_high,
                          probe_channel_num, 
                          channel_path, 
                          vad_half_window, 
                          label_save, 
                          wav_save, 
                          seeg_save, 
                          is_VAD, 
                          bef_plot, 
                          aft_plot, 
                          env_plot,
                          wav_plot, 
                          save_mat):
    
    """Data processing for speech activity detection

    Parameters
    -----------
    system:
        The data acquisition system, either Natus or Neuracle
    raw_root_path: 
        Path to load the raw dataset
    processed_root_path: 
        Path to store the processed dataset
    paradigm:
        Experimental paradigm version
    subject_list: 
        List of subjects
    segment_list:
        List of segments for data storage, specific to the Neuracle system
    session_list: 
        List of experimental sessions
    channel_list:
        List of channels
    single_channel_list: 
        List of single electrodes
    bad_channel_list:
        List of bad channels
    env_sample_rate:
        Sampling rate after envelope downsampling
    wav_sample_rate:
        Sampling rate after audio downsampling
    f_low: 
        Low-frequency cutoff for bandpass filtering
    f_high: 
        High-frequency cutoff for bandpass filtering
    probe_channel_num:
        Number of contacts on a single electrode when fully loaded
    channel_path:
        Data storage version: 'read_good_single' for single electrodes; 'read_good' for multi-electrode setup
    vad_half_window:
        Half-window length for voice activity detection (VAD)
    label_save:
        Whether to save labels
    wav_save:
        Whether to save processed audio data
    seeg_save:
        Whether to save processed SEEG data
    is_VAD:
        Whether to perform voice activity detection (VAD)
    bef_plot:
        Whether to plot data before processing (raw data plot)
    aft_plot:
        Whether to plot data after processing (processed data plot, before envelope extraction)
    env_plot:
        Whether to plot the envelope
    wav_plot:
        Whether to plot audio data
    save_mat:
        Whether to store data in .mat format for use with MATLAB's acoustic contamination check toolkit
    """
    
    for subject_id in range(len(subject_list)):
        for i in range(session_list[subject_id]):
            label_path = raw_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/label_read/session{i+1}'
            wav_path = raw_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/audio/session{i+1}'
            seeg_path = sessionFolderFind(raw_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/seeg', i+1)

            label_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/trans_label'  
            wav_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/wav/audio_vad'
            wav_mat_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/{channel_path}/wav_mat'
            
            seeg_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/{channel_path}/seeg_vad'
            seeg_mat_save_path = processed_root_path + f'{system}/{paradigm}/{subject_list[subject_id]}/read/session{i+1}/{channel_path}/seeg_mat'
            

            # label
            preprocessor = PreProcessor(block_num=10, trial_num=48, trial_len=1.6)
            preprocessor.readLabel(label_path, label_save_path, save=label_save)

            # wav
            preprocessor.wavSplit(wav_path, 
                                  wav_save_path, 
                                  wav_mat_save_path, 
                                  new_sample_rate=wav_sample_rate, 
                                  save=wav_save, 
                                  plot=wav_plot, 
                                  is_VAD=is_VAD, 
                                  save_mat=save_mat, 
                                  vad_half_window=vad_half_window)
            print('preprocessor.wav_cross.shape:', preprocessor.wav_rest.shape)
            print('preprocessor.wav_word.shape:', preprocessor.wav_word.shape)


            # seeg
            if 'single' in channel_path:
                preprocessor.readSEEG(system, 
                                      seeg_path, 
                                      seeg_mat_save_path, 
                                      segment_list[subject_id][i], 
                                      single_channel_list[subject_list[subject_id]], 
                                      bad_channel_list[subject_list[subject_id]], 
                                      plot=bef_plot, 
                                      save_mat=save_mat)
            else:
                preprocessor.readSEEG(system, 
                                      seeg_path, 
                                      seeg_mat_save_path, 
                                      segment_list[subject_id][i], 
                                      channel_list[subject_list[subject_id]], 
                                      bad_channel_list[subject_list[subject_id]], 
                                      plot=bef_plot, 
                                      save_mat=save_mat)

            preprocessor.dataDetrend()

            if 'bipolar' in channel_path:
                preprocessor.reReference_Bipolar(probe_channel_num)
                print("reReference_Bipolar apply")
            else:
                preprocessor.reReference()
                print("reReference apply")
            
            preprocessor.filter(f_low=f_low, f_high=f_high)
            preprocessor.clamping(plot=aft_plot)
            preprocessor.hilbertTrans(plot=env_plot)
            preprocessor.dataResample(env_sample_rate)

            preprocessor.dataSplit(seeg_save_path, 
                                   env_sample_rate, 
                                   save=seeg_save, 
                                   is_VAD=is_VAD, 
                                   vad_half_window=vad_half_window)
            print('preprocessor.envelope_word.shape:', preprocessor.envelope_word.shape)
            print('preprocessor.envelope_rest.shape:', preprocessor.envelope_rest.shape)

    if label_save == False:
        print("label not saved !!!")
    if wav_save == False:
        print("wav not saved !!!")
    if seeg_save == False:
        print("seeg not saved !!!")



if __name__ == '__main__':

    system = 'Neuracle' # Natus | Neuracle
    process_type = 'word' # detect | word

    channel_path_list = ['read_full',
                         'read_single_A',
                         'read_single_B',
                         'read_single_C',
                         'read_single_D',
                         'read_single_E',
                         'read_single_F',
                         'read_single_G',
                         'read_full_bipolar',
                         'read_single_A_bipolar',
                         'read_single_B_bipolar',
                         'read_single_C_bipolar',
                         'read_single_D_bipolar',
                         'read_single_E_bipolar',
                         'read_single_F_bipolar',
                         'read_single_G_bipolar']

    if system == 'Natus':
        config = Natus_Config()
        single_channel_dict_list = [{'S01':[],
                                     'S02':[],
                                     'S03':[],
                                     'S04':[],
                                     'S05':[],
                                     'S06':[],
                                     'S09':[]},
                                    {'S01':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S02':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S03':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S04':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S05':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S06':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S09':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']},
                                    {'S01':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S02':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S03':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S04':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S05':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S06':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S09':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']},
                                    {'S01':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S02':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S03':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S04':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S05':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S06':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S09':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']},
                                    {'S01':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S02':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S03':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S04':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S05':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S06':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S09':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']},
                                    {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                     'S02':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S03':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S04':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S05':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S06':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S09':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']},
                                    {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                     'S02':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S03':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S04':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S05':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S06':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S09':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']},
                                    {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                     'S02':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S03':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S04':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S05':['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'],
                                     'S06':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S09':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']},
                                    {'S01':[],
                                     'S02':[],
                                     'S03':[],
                                     'S04':[],
                                     'S05':[],
                                     'S06':[],
                                     'S09':[]},
                                    {'S01':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S02':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S03':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S04':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S05':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S06':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S09':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']},
                                    {'S01':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S02':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S03':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S04':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S05':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S06':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S09':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']},
                                    {'S01':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S02':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S03':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S04':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S05':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S06':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S09':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']},
                                    {'S01':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S02':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S03':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S04':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S05':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S06':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S09':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']},
                                    {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                     'S02':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S03':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S04':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S05':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S06':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S09':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']},
                                    {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                     'S02':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S03':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S04':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S05':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S06':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S09':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']},
                                    {'S01':['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                                     'S02':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S03':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S04':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S05':['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'],
                                     'S06':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S09':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']}]
        
    elif system == 'Neuracle':
        config = Neuracle_Config()
        single_channel_dict_list = [{'S07':[],
                                     'S08':[],
                                     'S10':[]},
                                    {'S07':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S08':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S10':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']},
                                    {'S07':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S08':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S10':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']},
                                    {'S07':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S08':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S10':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']},
                                    {'S07':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S08':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S10':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']},
                                    {'S07':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S08':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S10':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']},
                                    {'S07':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S08':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S10':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']},
                                    {'S07':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S08':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S10':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']},
                                    {'S07':[],
                                     'S08':[],
                                     'S10':[]},
                                    {'S07':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S08':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
                                     'S10':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']},
                                    {'S07':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S08':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
                                     'S10':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']},
                                    {'S07':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S08':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                                     'S10':['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']},
                                    {'S07':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S08':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                                     'S10':['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']},
                                    {'S07':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S08':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'],
                                     'S10':['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']},
                                    {'S07':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S08':['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
                                     'S10':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']},
                                    {'S07':['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
                                     'S08':['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
                                     'S10':['F1', 'F2', 'F3', 'F4', 'F5', 'F6']}]

    if process_type == 'word':
        for channel_path, single_channel_list in zip(channel_path_list, single_channel_dict_list):
            config.channel_path = channel_path
            config.single_channel_list = single_channel_list
            run_preprocess_word(system, 
                                config.raw_root_path, 
                                config.processed_root_path, 
                                config.paradigm, 
                                config.subject_list, 
                                config.segment_list, 
                                config.session_list, 
                                config.channel_list, 
                                config.single_channel_list, 
                                config.bad_channel_list, 
                                config.env_sample_rate, 
                                config.wav_sample_rate, 
                                config.f_low, 
                                config.f_high, 
                                config.probe_channel_num, 
                                config.channel_path, 
                                config.vad_half_window, 
                                config.label_save, 
                                config.wav_save, 
                                config.seeg_save, 
                                config.is_VAD, 
                                config.bef_plot, 
                                config.aft_plot, 
                                config.env_plot, 
                                config.wav_plot, 
                                config.save_mat)

    elif process_type == 'detect':
        config.is_VAD = True
        for channel_path, single_channel_list in zip(channel_path_list, single_channel_dict_list):
            config.channel_path = channel_path
            config.single_channel_list = single_channel_list
            run_preprocess_detect(system,
                                  config.raw_root_path, 
                                  config.processed_root_path, 
                                  config.paradigm, 
                                  config.subject_list, 
                                  config.segment_list, 
                                  config.session_list, 
                                  config.channel_list, 
                                  config.single_channel_list, 
                                  config.bad_channel_list, 
                                  config.env_sample_rate, 
                                  config.wav_sample_rate, 
                                  config.f_low, 
                                  config.f_high, 
                                  config.probe_channel_num, 
                                  config.channel_path, 
                                  config.vad_half_window, 
                                  config.label_save, 
                                  config.wav_save, 
                                  config.seeg_save, 
                                  config.is_VAD, 
                                  config.bef_plot, 
                                  config.aft_plot, 
                                  config.env_plot, 
                                  config.wav_plot, 
                                  config.save_mat) 