# Dataset class
class SEEDVII_Dataset(Dataset):
    
    def __init__(self, data_dir: str = ".", modality: str = 'multimodal', 
                 subset_ratio: float = 0.01):
        self.data_dir = data_dir
        self.modality = modality
        self.subset_ratio = subset_ratio
        
        # Dataset specifications
        self.num_classes = 7
        self.num_subjects = 20
        self.eeg_feature_dim = 310
        self.eye_feature_dim = 33
        
        # Load and process data
        self._load_data()
        if self.subset_ratio < 1.0:
            self._create_subset()
    
    def _load_data(self):
        """Loading EEG and eye movement features here."""
        print(f"Loading SEED-VII dataset from {self.data_dir}")
        
        eeg_data, eye_data = [], []
        emotion_labels, subject_labels = [], []
        
        # Get feature files
        eeg_dir = os.path.join(self.data_dir, 'EEG_features')
        eye_dir = os.path.join(self.data_dir, 'EYE_features')
        
        eeg_files = sorted(glob.glob(os.path.join(eeg_dir, '*.mat')))
        eye_files = sorted(glob.glob(os.path.join(eye_dir, '*.mat')))
        
        print(f"Found {len(eeg_files)} EEG files and {len(eye_files)} eye files")
        
        # Simple emotion mapping (based on paper structure)
        emotion_map = self._get_emotion_mapping()
        
        # Process subjects
        for subject_idx, eeg_file in enumerate(eeg_files[:self.num_subjects]):
            subject_name = os.path.basename(eeg_file).replace('.mat', '')
            print(f"Loading subject {subject_idx + 1}: {subject_name}")
            
            # Load files
            try:
                eeg_mat = sio.loadmat(eeg_file)
                eye_file = self._find_matching_eye_file(subject_name, eye_files)
                if eye_file:
                    eye_mat = sio.loadmat(eye_file)
                else:
                    continue
            except Exception as e:
                print(f"Error loading files for {subject_name}: {e}")
                continue
            
            # Process videos
            for video_id in range(1, 81):
                eeg_features, eye_features = self._extract_features(
                    eeg_mat, eye_mat, video_id)
                
                if eeg_features is not None and eye_features is not None:
                    min_windows = min(len(eeg_features), len(eye_features))
                    if min_windows > 0:
                        eeg_data.append(eeg_features[:min_windows])
                        eye_data.append(eye_features[:min_windows])
                        
                        emotion_label = emotion_map.get(video_id, 6)
                        emotion_labels.extend([emotion_label] * min_windows)
                        subject_labels.extend([subject_idx] * min_windows)
        
        # Convert to arrays
        self.eeg_features = np.vstack(eeg_data)
        self.eye_features = np.vstack(eye_data)
        self.emotion_labels = np.array(emotion_labels)
        self.subject_labels = np.array(subject_labels)
        
        print(f"Dataset loaded: {len(self.emotion_labels)} samples")
        print(f"EEG shape: {self.eeg_features.shape}, Eye shape: {self.eye_features.shape}")
    
    def _get_emotion_mapping(self):
        """Here I tried to replicate a simple emotion mapping from the MAET paper."""
        emotion_map = {}
        # Simplified mapping - 4 videos per emotion per session
        for session in range(4):
            emotions = [0, 6, 3, 1, 5, 2, 4][0:7] if session % 2 == 0 else [5, 1, 2, 6, 0, 4, 3]
            for i, emotion in enumerate(emotions):
                for video in range(4):
                    video_id = session * 20 + i * 4 + video + 1
                    if video_id <= 80:
                        emotion_map[video_id] = emotion % 7
        return emotion_map
    
    def _find_matching_eye_file(self, subject_name, eye_files):
        """This code is to find the matching eye file for the subjects."""
        for eye_file in eye_files:
            if subject_name in os.path.basename(eye_file):
                return eye_file
        return None
    
    def _extract_features(self, eeg_mat, eye_mat, video_id):
        video_key = str(video_id)
        
        # Try different key formats for EEG
        eeg_features = None
        for key in [f'de_LDS_{video_id}', f'de_{video_id}', video_key]:
            if key in eeg_mat:
                eeg_features = eeg_mat[key]
                break
        
        # Try different key formats for Eye
        eye_features = None
        for key in [video_key, str(video_id)]:
            if key in eye_mat:
                eye_features = eye_mat[key]
                break
        
        # Process EEG features
        if eeg_features is not None:
            if eeg_features.ndim == 3:
                eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
            if eeg_features.shape[1] != self.eeg_feature_dim:
                eeg_features = None
        
        # Process eye features
        if eye_features is not None and eye_features.shape[1] != self.eye_feature_dim:
            eye_features = None
        
        return eeg_features, eye_features
    
    def _create_subset(self):
        n_samples = len(self.emotion_labels)
        subset_size = max(1, int(n_samples * self.subset_ratio))
        
        try:
            indices = np.arange(n_samples)
            subset_indices, _ = train_test_split(
                indices, train_size=subset_size, stratify=self.emotion_labels, random_state=42)
        except:
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
        
        self.eeg_features = self.eeg_features[subset_indices]
        self.eye_features = self.eye_features[subset_indices]
        self.emotion_labels = self.emotion_labels[subset_indices]
        self.subject_labels = self.subject_labels[subset_indices]
        
        print(f"Created {self.subset_ratio*100:.1f}% subset: {len(self.emotion_labels)} samples")
    
    def __len__(self):
        return len(self.emotion_labels)
    
    def __getitem__(self, idx):
        sample = {}
        
        if self.modality in ['eeg', 'multimodal']:
            sample['eeg'] = torch.FloatTensor(self.eeg_features[idx])
        if self.modality in ['eye', 'multimodal']:
            sample['eye'] = torch.FloatTensor(self.eye_features[idx])
        
        sample['label'] = torch.LongTensor([self.emotion_labels[idx]])[0]
        sample['subject'] = torch.LongTensor([self.subject_labels[idx]])[0]
        
        return sample
