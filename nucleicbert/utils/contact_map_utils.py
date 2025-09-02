import torch
import numpy as np


def distance_to_binary(self, tensor, cutoff):
    """
    Convert a distance class tensor to a binary tensor based on the provided cutoff,
    while converting class 20 to 0.

    Parameters:
    - tensor (torch.Tensor): The input tensor with distance classes.
    - cutoff (float): The distance cutoff to classify as 1. Distances <= cutoff are 1, otherwise 0.
    - class_intervals (list of tuples): List of (start, end) tuples for each class interval.
    
    Returns:
    - binary_tensor (torch.Tensor): The binary tensor.
    """
    class_intervals = [
    (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), 
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), 
    (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, float('inf'))
    ]
    # Create the binary mapping based on the cutoff
    binary_mapping = torch.tensor([1 if interval[1] <= cutoff else 0 for interval in class_intervals], dtype=torch.float32, device=self.device)
    
    # Ensure there's a mapping for class 20
    if len(binary_mapping) == 20:
        binary_mapping = torch.cat((binary_mapping, torch.tensor([0], device=self.device)), 0)

    # Apply the mapping to convert the distance classes to binary values
    binary_tensor = binary_mapping[tensor]

    return binary_tensor.to(torch.bool)

def logits_to_binary_predictions(self, logits, cutoff):
    """
    Convert logits for distance classes to binary predictions based on the provided cutoff.

    Parameters:
    - logits (torch.Tensor): The input logits for distance classes with shape (B, C, L, L).
    - cutoff (float): The distance cutoff to classify as 1. Distances <= cutoff are 1, otherwise 0.

    Returns:
    - binary_preds (torch.Tensor): The binary predictions with shape (B, 1, L, L).
    """
    class_intervals = [
        (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), 
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), 
        (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), 
        (20, float('inf'))
    ]
    
    # Create the binary mapping based on the cutoff
    binary_mapping = torch.tensor([1 if interval[1] <= cutoff else 0 for interval in class_intervals], dtype=torch.float32, device=self.device)
    
    # Ensure there's a mapping for class 20 (NaN values)
    if len(binary_mapping) == 20:
        binary_mapping = torch.cat((binary_mapping, torch.tensor([0], device=self.device)), 0)

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)  # Apply softmax on the class dimension (C)

    # Apply the binary mapping to convert class probabilities to binary probabilities
    binary_mapping = binary_mapping.long()
    
    # Sum probabilities for classes mapped to 0 and 1
    prob_0 = probs[:, binary_mapping == 0, :, :].sum(dim=1)
    prob_1 = probs[:, binary_mapping == 1, :, :].sum(dim=1)
    
    # Compute the final binary probability tensor
    binary_probs = torch.stack((prob_0, prob_1), dim=1)

    # Get the probabilities for class 1
    prob_1 = binary_probs[:, 1, :, :]

    # Apply the threshold to get binary predictions
    # binary_preds = (prob_1 > 0.5).float().unsqueeze(1)

    return prob_1



class ContactProbabilityCalculator:
    def __init__(self, device='cuda', is_backbone_class=False):
        self.device = device
        # Define the class intervals explicitly
        self.class_intervals = [
            (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), 
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), 
            (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, float('inf'))
        ]
        self.is_backbone_class = is_backbone_class
        
    def compute_contact_probabilities(self, logits, cutoff=8.0):
        """
        Compute contact probabilities from distance class logits.
        
        Args:
            logits (torch.Tensor): Shape (batch_size, 20, seq_len, seq_len) containing logits for each distance class
            cutoff (float): Distance cutoff in Angstroms to define a contact
            
        Returns:
            torch.Tensor: Contact probabilities of shape (batch_size, seq_len, seq_len)
        """
        # First get probabilities for each class using softmax
        # Shape: (batch_size, 20, seq_len, seq_len)
        if self.is_backbone_class:
            logits = logits[:,:20,:,:]

        class_probs = torch.softmax(logits, dim=1)
        
        # Create a mask for which classes correspond to contacts
        # A class is considered a contact if its upper bound is <= cutoff
        contact_class_mask = torch.zeros(len(self.class_intervals), device=self.device)
        for i, (_, upper_bound) in enumerate(self.class_intervals):
            if upper_bound <= cutoff:
                contact_class_mask[i] = 1.0
                
        # Reshape mask for broadcasting
        # Shape: (1, 20, 1, 1)
        contact_class_mask = contact_class_mask.view(1, -1, 1, 1)
        
        # Sum probabilities of all classes that represent contacts
        # This gives us the total probability of the distance being <= cutoff
        contact_probs = (class_probs * contact_class_mask).sum(dim=1)
        
        return contact_probs
    
    def compute_detailed_probabilities(self, logits, cutoff=8.0):
        """
        Compute detailed probability analysis including class-wise contributions.
        
        Args:
            logits (torch.Tensor): Shape (batch_size, 20, seq_len, seq_len)
            cutoff (float): Distance cutoff
            
        Returns:
            dict: Detailed probability analysis
        """
        if self.is_backbone_class:
            logits = logits[:,:20,:,:]
        class_probs = torch.softmax(logits, dim=1)
        batch_size, num_classes, seq_len, _ = class_probs.shape
        
        # Store individual class probabilities
        class_contributions = {}
        for i, (lower, upper) in enumerate(self.class_intervals):
            class_contributions[f'class_{i}_({lower}-{upper})'] = class_probs[:, i]
        
        # Calculate cumulative probabilities up to each distance
        cumulative_probs = torch.zeros((batch_size, len(self.class_intervals), seq_len, seq_len), 
                                     device=self.device)
        for i in range(len(self.class_intervals)):
            cumulative_probs[:, i] = class_probs[:, :i+1].sum(dim=1)
            
        # Get total contact probability
        contact_probs = self.compute_contact_probabilities(logits, cutoff)
        
        return {
            'class_probabilities': class_probs,  # Individual class probabilities
            'cumulative_probabilities': cumulative_probs,  # Cumulative probabilities
            'contact_probability': contact_probs,  # Final contact probability
            'class_contributions': class_contributions  # Individual class contributions
        }
    
    def get_expected_distance(self, logits):
        """
        Calculate expected distance for each position.
        
        Args:
            logits (torch.Tensor): Shape (batch_size, 20, seq_len, seq_len)
            
        Returns:
            torch.Tensor: Expected distances
        """
        if self.is_backbone_class:
            logits = logits[:,:20,:,:]
        class_probs = torch.softmax(logits, dim=1)
        
        # Calculate midpoint of each distance bin
        class_distances = torch.tensor(
            [(start + end) / 2 for start, end in self.class_intervals[:-1]] + [20.0],  # Use 20Å for last bin
            device=self.device
        ).view(1, -1, 1, 1)
        
        # Calculate expected distance as weighted sum
        expected_distance = (class_probs * class_distances).sum(dim=1)
        
        return expected_distance

# Example usage:
"""
calculator = DetailedContactProbabilityCalculator(device='cuda')

# Assuming logits has shape (batch_size, 20, seq_len, seq_len)
detailed_probs = calculator.compute_detailed_probabilities(logits, cutoff=8.0)

# Get various probability measures
contact_probs = detailed_probs['contact_probability']  # Overall contact probability
class_probs = detailed_probs['class_probabilities']   # Individual class probabilities
cumulative = detailed_probs['cumulative_probabilities']  # Cumulative probabilities

# Get expected distances
expected_distances = calculator.get_expected_distance(logits)
"""



class DistanceToContactConverter:
    def __init__(self, device='cuda', is_backbone_class=False):
        self.device = device
        self.is_backbone_class = is_backbone_class
        
    def distance_to_binary_contacts(self, distances, cutoff, min_sep=None, max_sep=None):
        """
        Convert distance matrix to binary contact matrix.
        
        Args:
            distances (torch.Tensor or np.ndarray): Distance matrix of shape (batch_size, L, L) or (L, L)
            cutoff (float): Distance cutoff in Angstroms. Distances <= cutoff are considered contacts
            min_sep (int, optional): Minimum sequence separation. Defaults to None
            max_sep (int, optional): Maximum sequence separation. Defaults to None
            
        Returns:
            torch.Tensor: Binary contact matrix of same shape as input
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(distances, np.ndarray):
            distances = torch.from_numpy(distances)
        
        # Move to specified device
        distances = distances.to(self.device)
        
        # Add batch dimension if not present
        if distances.dim() == 2:
            distances = distances.unsqueeze(0)
            
        # Create binary contact map
        contacts = (distances <= cutoff).to(torch.float32)
        
        # Apply sequence separation masks if specified
        if min_sep is not None or max_sep is not None:
            batch_size, seq_len, _ = distances.shape
            seq_positions = torch.arange(seq_len, device=self.device)
            sequence_sep = torch.abs(seq_positions.unsqueeze(0) - seq_positions.unsqueeze(1))
            sequence_sep = sequence_sep.unsqueeze(0).expand(batch_size, -1, -1)
            
            if min_sep is not None:
                contacts = contacts * (sequence_sep >= min_sep).float()
            
            if max_sep is not None:
                contacts = contacts * (sequence_sep <= max_sep).float()
        
        return contacts
    
    def class_distances_to_binary(self, class_distances, cutoff=8.0, class_boundaries=None):
        """
        Convert distance class indices to binary contacts.
        
        Args:
            class_distances (torch.Tensor): Tensor of distance class indices
            cutoff (float): Distance cutoff in Angstroms
            class_boundaries (list, optional): List of class boundaries. 
                Defaults to standard boundaries used in protein distance prediction
                
        Returns:
            torch.Tensor: Binary contact matrix
        """
        if class_boundaries is None:
            class_boundaries = [
                (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
                (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, float('inf'))
            ]
            
        # Convert to tensor if not already
        if isinstance(class_distances, np.ndarray):
            class_distances = torch.from_numpy(class_distances)
            
        class_distances = class_distances.to(self.device)
        
        # Create mapping from class indices to binary contacts
        num_classes = len(class_boundaries)
        if self.is_backbone_class:
            num_classes += 1
        class_is_contact = torch.zeros(num_classes, device=self.device)
        
        for i, (_, upper_bound) in enumerate(class_boundaries):
            if upper_bound <= cutoff:
                class_is_contact[i] = 1
                
        # Handle special case for last class (usually represents distances > 20Å)
        class_is_contact[-1] = 0
        
        # Map class indices to binary contacts
        return class_is_contact[class_distances].bool()
    
    def validate_contacts(self, contacts):
        """
        Validate properties of contact matrix.
        
        Args:
            contacts (torch.Tensor): Binary contact matrix
            
        Returns:
            dict: Validation results
        """
        results = {
            "is_binary": torch.all(torch.logical_or(contacts == 0, contacts == 1)),
            "is_symmetric": torch.all(contacts == contacts.transpose(-2, -1))
        }
        
        # Check contact density
        contact_density = contacts.float().mean().item()
        results["contact_density"] = contact_density
        
        return results

# Example usage:
"""
converter = DistanceToContactConverter(device='cuda')

# Convert raw distances to contacts
distances = torch.randn(1, 100, 100)  # Example distance matrix
contacts = converter.distance_to_binary_contacts(
    distances, 
    cutoff=8.0, 
    min_sep=6,  # Minimum sequence separation
    max_sep=None  # No maximum separation
)

# Convert distance classes to contacts
class_distances = torch.randint(0, 20, (1, 100, 100))  # Example class indices
class_contacts = converter.class_distances_to_binary(
    class_distances,
    cutoff=8.0
)

# Validate results
validation = converter.validate_contacts(contacts)
print("Validation results:", validation)
"""
