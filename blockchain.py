import hashlib
import json
from time import time
from datetime import datetime


class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_data = []
        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash=None):
        """
        Create a new Block in the Blockchain
        :param proof: <int> The proof given by the Proof of Work algorithm
        :param previous_hash: (Optional) <str> Hash of previous Block
        :return: <dict> New Block
        """

        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.now()),
            'data': self.pending_data,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of pending data
        self.pending_data = []

        self.chain.append(block)
        return block

    @property
    def last_block(self):
        return self.chain[-1]

    def new_data(self, data_type, data_payload):
        """
        Adds new data to the list of pending data
        :param data_type: <str> The type of data (e.g., 'crash', 'speed', 'vehicle_info')
        :param data_payload: <dict> The data itself
        :return: <int> The index of the Block that will hold this data
        """
        self.pending_data.append({
            'data_type': data_type,
            'payload': data_payload
        })
        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        :param block: <dict> Block
        :return: <str>
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_proof):
        """
        Simple Proof of Work Algorithm:
         - Find a number 'proof' such that hash(last_proof, proof) contains 4 leading zeroes
        :param last_proof: <int>
        :return: <int>
        """
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1

        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        """
        Validates the proof: Does hash(last_proof, proof) contain 4 leading zeroes?
        :param last_proof: <int> Previous Proof
        :param proof: <int> Current Proof
        :return: <bool> True if correct, False if not.
        """
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def save_chain(self, path='blockchain.json'):
        """Saves the blockchain to a file."""
        with open(path, 'w') as f:
            json.dump(self.chain, f, indent=4)

    def load_chain(self, path='blockchain.json'):
        """Loads the blockchain from a file."""
        try:
            with open(path, 'r') as f:
                self.chain = json.load(f)
        except FileNotFoundError:
            pass  # Chain file doesn't exist yet, will be created on first save


# Instantiate the Blockchain
blockchain = Blockchain()
blockchain.load_chain()


def add_to_blockchain(data_type, data_payload):
    """Public function to add data to the blockchain and save it."""
    blockchain.new_data(data_type, data_payload)

    # Mine a new block
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    # Forge the new Block by adding it to the chain
    previous_hash = blockchain.hash(last_block)
    blockchain.new_block(proof, previous_hash)

    # Save the updated chain
    blockchain.save_chain()
    print(f"Added to blockchain: {data_type}")


if __name__ == "__main__":
    from pprint import pprint

    # Example data add
    add_to_blockchain("crash", {"location": "Delhi", "time": str(datetime.now())})
    add_to_blockchain("speed", {"vehicle_id": "MH12AB1234", "speed": 80})

    print("\n✅ Blockchain Updated! Current Chain:")
    pprint(blockchain.chain, indent=4)































# import hashlib
# import json
# from datetime import datetime
# import threading
# import os
#
# # Thread lock for safe concurrent access
# lock = threading.Lock()
#
#
# class Blockchain:
#     def __init__(self, path="blockchain.json"):
#         self.chain = []
#         self.pending_data = []
#         self.path = path
#
#         # Try loading existing chain, else create genesis
#         self.load_chain()
#
#     def create_genesis_block(self):
#         """Creates the first block in the chain (genesis)."""
#         block = {
#             "index": 1,
#             "timestamp": str(datetime.now()),
#             "data": "Genesis Block",
#             "proof": 100,
#             "previous_hash": "1",
#         }
#         self.chain.append(block)
#         return block
#
#     def new_block(self, proof, previous_hash=None):
#         """Create a new Block in the Blockchain."""
#         block = {
#             "index": len(self.chain) + 1,
#             "timestamp": str(datetime.now()),
#             "data": self.pending_data,
#             "proof": proof,
#             "previous_hash": previous_hash or self.hash(self.chain[-1]),
#         }
#
#         # Reset pending data
#         self.pending_data = []
#         self.chain.append(block)
#         return block
#
#     @property
#     def last_block(self):
#         return self.chain[-1]
#
#     def new_data(self, data_type, data_payload):
#         """Add new data to the list of pending data."""
#         self.pending_data.append({
#             "data_type": data_type,
#             "payload": data_payload,
#         })
#         return self.last_block["index"] + 1
#
#     @staticmethod
#     def hash(block):
#         """Creates a SHA-256 hash of a Block."""
#         block_string = json.dumps(block, sort_keys=True).encode()
#         return hashlib.sha256(block_string).hexdigest()
#
#     def proof_of_work(self, last_proof):
#         """Simple Proof of Work Algorithm."""
#         proof = 0
#         while not self.valid_proof(last_proof, proof):
#             proof += 1
#         return proof
#
#     @staticmethod
#     def valid_proof(last_proof, proof):
#         """Validates the proof: Does hash(last_proof, proof) contain 4 leading zeroes?"""
#         guess = f"{last_proof}{proof}".encode()
#         guess_hash = hashlib.sha256(guess).hexdigest()
#         return guess_hash[:4] == "0000"
#
#     def save_chain(self):
#         """Saves the blockchain to a file."""
#         with open(self.path, "w") as f:
#             json.dump(self.chain, f, indent=4)
#
#     def load_chain(self):
#         """Loads the blockchain from a file, with fallback to genesis."""
#         try:
#             if not os.path.exists(self.path):
#                 print("⚠️ No blockchain file found. Creating genesis block.")
#                 self.create_genesis_block()
#                 self.save_chain()
#                 return
#
#             with open(self.path, "r") as f:
#                 content = f.read().strip()
#                 if not content:  # Empty file
#                     print("⚠️ Blockchain file empty. Creating genesis block.")
#                     self.create_genesis_block()
#                     self.save_chain()
#                 else:
#                     self.chain = json.loads(content)
#
#         except json.JSONDecodeError:
#             print("⚠️ Blockchain file corrupted. Resetting with genesis block.")
#             self.chain = []
#             self.create_genesis_block()
#             self.save_chain()
#
#
# # Global blockchain instance
# blockchain = Blockchain()
#
#
# def add_to_blockchain(data_type, data_payload):
#     """Public function to add data safely to the blockchain and save it."""
#     with lock:  # Thread-safe section
#         blockchain.new_data(data_type, data_payload)
#
#         # Mine a new block
#         last_block = blockchain.last_block
#         last_proof = last_block["proof"]
#         proof = blockchain.proof_of_work(last_proof)
#
#         previous_hash = blockchain.hash(last_block)
#         blockchain.new_block(proof, previous_hash)
#
#         blockchain.save_chain()
#         print(f"✅ Added to blockchain: {data_type}")
#
#
# # Demo usage
# if __name__ == "__main__":
#     from pprint import pprint
#
#     add_to_blockchain("crash", {"location": "Delhi", "time": str(datetime.now())})
#     add_to_blockchain("speed", {"vehicle_id": "MH12AB1234", "speed": 80})
#
#     print("\n✅ Blockchain Updated! Current Chain:")
#     pprint(blockchain.chain, indent=4)
