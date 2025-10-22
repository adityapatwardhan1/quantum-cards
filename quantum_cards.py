# THIS IS A CLI VERSION IT IS BETTER

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, random_unitary
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt

# TODO: 
# 1. Full Grover oracle operator (make it work on a partial bitstring)
# 2. Comment the code


"""
Quantum cards with measurement
1. There are n (3 to 7, inclusive) qubits, each starting in the |+> state
2. Each player is assigned bitstrings of length n, which, if it is a measurement
outcome, earns them points.
Player with the most points after running 1024 shots of measurements wins.
3. Each player has some cards that correspond to unitary gates
They may pick which qubits it operates on, however
4. Special cards besides the standard unitary gates we learned in class:
Diffusion operator card: applies the diffusion operator from Grover's which can
help if they have a good chance of their bitstring being measured
- Perhaps make the diffusion only work on some qubits or instead of a full
"flip over the average" you apply a gate besides CZ to do a "partial diffusion"
Reverse card: Undo the other player's last move
Reset a qubit to zero
5. Once there are no more cards, measure the outcome and assign points.

How this meets the requirements:
1. Superposition: The qubits begin in the |+> state and the quantum state is
a superposition
2. Entanglement: Can be achieved through controlled gates. Useful for Grover
oracles and useful strategically if one has strings like 000 and 110 where
one can use CNOT to entangle the 2 leftmost qubits.
3. Measurement: obvious
4. Interference: Amplitude amplification/Grover, terms can cancel in the math
of the game plausibly
"""

statevector_plot_fig = None
statevector_plot_axes = None

MOVE_NAMES = [
    'X', 'Y', 'Z', 'H', 'CX', 'CCX', 'I', 'S', 'S_dag', 'T', 'T_dag',
    'SWAP', 'DIFFUSION', 'CZ', 'CCZ', 'RESET', 'GROVER'
]

MOVE_WEIGHTS = [
    3, 1, 2, 4, 3, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 3
]

MOVE_TO_NUM_QUBITS = {
    'X': 1, 'Y': 1, 'Z': 1, 'H': 1, 'CX': 2, 'CCX': 3, 'I': 0, 'S': 1, 'S_dag': 1,
    'T': 1, 'T_dag': 1, 'SWAP': 2, 'CZ': 2, 'CCZ': 3, 'DIFFUSION': 0, 'RESET': 1,
    'GROVER': 0
}

def generate_bitstring(length: int):
  """Generates a random bitstring of length n. Used for generating strings
  at the beginning of the game for each player to aim for as a measurement
  outcome."""
  bitstring = ""
  for _ in range(length):
    bitstring += random.choice(['0', '1'])
  return bitstring

def get_qubit_number(num_qubits: int):
    """Reads & validates input for what qubit the user wants the operation
    to perform on"""
    qubit_num = ""
    while True:
      try:
        print(f"Enter a number between 0 and {num_qubits - 1} " +
          "inclusive")
        the_input = input()
        qubit_num = int(the_input)
        if qubit_num < 0 or qubit_num >= num_qubits:
          continue
        else:
          break
      except ValueError:
        print("Not a valid integer input")
    return qubit_num

def get_n_qubit_numbers(num_qubits: int, n: int):
  """Reads input for what qubits the user wants operation to perform on
  The qubits involved in the operation must be distinct"""
  qubit_numbers = []
  for i in range(n):
    qubit_num = get_qubit_number(num_qubits)
    while qubit_num in qubit_numbers:
      print("You have already selected that qubit, try again")
      qubit_num = get_qubit_number(num_qubits)
    qubit_numbers.append(qubit_num)
  return qubit_numbers

class Card:
  """Class storing information about a card: operation and number of qubits"""
  def __init__(self, operation_name, num_qubits_involved):
    self.operation_name = operation_name
    self.num_qubits_involved = num_qubits_involved

  def __repr__(self):
    return self.operation_name

def generate_random_deck(num_cards, num_qubits_in_game):
  """Generates a random deck at the beginning of the game"""
  card_names = random.choices(MOVE_NAMES, MOVE_WEIGHTS, k=num_cards)
  cards = [Card(card_name, MOVE_TO_NUM_QUBITS.get(card_name, num_qubits_in_game)) for card_name in card_names]
  return cards

def get_deck_operation(card_deck: list):
  """Removes and returns a card from the deck based on user input"""
  card_num = -1
  # Input validation loop
  while card_num < 0 or card_num >= len(card_deck):
    try:
        print(f"Select a card from the deck by 0-based index: {card_deck}")
        the_input = input()
        card_num = int(the_input)
        if card_num < 0 or card_num >= len(card_deck):
          print("Not a valid index in this deck")
          continue
        else:
          break
    except ValueError:
      print("Not a valid integer input")
  return card_deck.pop(card_num)

def plot_statevector_data(statevector_dict, thresh_to_plot=0.02):
  """
  Plots a statevector as a bar graph
  :param statevector: Vector representing a quantum state
  :type statevector: list
  :param thresh_to_plot: Minimum magnitude for a statevector entry to be plotted.
    This reduces the number of entries to plot which is better visually.
  :type thresh_to_plot: float
  """
  global statevector_plot_axes, statevector_plot_fig
  if statevector_plot_fig is None or statevector_plot_axes is None:
        plt.ion()
        statevector_plot_fig, statevector_plot_axes = plt.subplots(figsize=(10, 5))
        statevector_plot_fig.canvas.manager.set_window_title("Statevector Probabilities")
  statevector_plot_axes.clear()

  keys = sorted(statevector_dict.keys())
  values = [statevector_dict[key] for key in keys]
  statevector_plot_axes.bar(keys, values, width=0.5, color='blue')
  statevector_plot_axes.set_ylabel('|Probability Amplitude|^2')
  statevector_plot_axes.set_xlabel('Measurement Outcome')
  statevector_plot_axes.set_xticklabels(keys, rotation=45)
  statevector_plot_axes.set_ylim([0, 1])

  statevector_plot_fig.tight_layout()
  statevector_plot_fig.canvas.draw()
  statevector_plot_fig.canvas.flush_events()
  plt.pause(0.1)
  pass

def plot_counts(counts, thresh_to_plot=0.02):
  """
  Plots a statevector as a bar graph
  :param counts: Vector representing counts after running a quantum circuit
  :type counts: dict
  :param thresh_to_plot: Minimum fraction of the times that a measurement
    outcome must occur among all the measurement outcomes to be plotted.
    This reduces the number of entries to plot which is better visually.
  :type thresh_to_plot: float
  """
  pass

def print_game_rules():
    rules = """
Welcome to QARDS! Here's how to play:\n
1. A quantum circuit where all qubits are in the |0> state is initialized. This number of qubits is at least 3.
2. Each player is assigned a number of bitstrings at the beginning of the game. The goal for a player is to increase the likelihood of measuring one of their bitstrings.
3. Cards are dealt at the beginning of the game randomly. There are different probabilities of dealing associated with each card being dealt.
4. A random unitary is applied to the original quantum circuit.
5. On each turn, a player chooses a card in their deck which corresponds to a unitary operator (except for the RESET card.) The player may choose which qubit(s) to apply this operator to, except for the Identity gate and Diffusion (which is applied to all qubits.)
6. Players take alternating turns until both players are out of cards.
7. After each player has played all of their cards, the circuit is measured with 1024 shots. The player who has more of their target bitstrings measured wins the game, and in the unlikely case that both players have the same number of bitstrings measured, there is a tie.
Enter any input when you are ready to proceed: 
"""
    print(rules)

def clear_terminal():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

def verify_bitstring_of_length_n(bitstring: str, n: int):
  if len(bitstring) != n:
    return False
  for i in range(len(bitstring)):
    if not bitstring[i] in ['0', '1']:
      return False
  return True

def get_grover_input(num_qubits: int):
  cur = ""
  while not verify_bitstring_of_length_n(cur, num_qubits):
    cur = input(f"Enter a bitstring whose length is {num_qubits} that you want to amplify:")
  return cur

class Game:
  """Class that manages game loop and flow"""

  def __init__(self, num_qubits, num_cards, num_bitstrings_per_player):
    """
    Constructor for Game object
    :param num_qubits: Number of qubits in the circuit for the game
    :type num_qubits: int
    :param num_cards: Number of cards in each player's deck to begin the game
    :type num_cards: int
    :param num_bitstrings_per_player: Number of target bitstrings for each player
    :type num_bitstrings_per_player: int
    """

    # Initialize the quantum circuit with all qubits in the |+> state
    self.num_qubits = num_qubits
    self.q: QuantumRegister = QuantumRegister(num_qubits)
    self.c: ClassicalRegister = ClassicalRegister(num_qubits)
    self.qc: QuantumCircuit = QuantumCircuit(self.q, self.c)
    for _ in range(num_qubits):
      U = random_unitary(2**self.num_qubits)
      self.qc.unitary(U, self.q)
      #print(random_unitary(self.num_qubits))
      # self.qc.h(self.q[i])

    # Generate decks
    self.num_cards_per_player = num_cards
    self.deck_player_one = generate_random_deck(num_cards, num_qubits)
    self.deck_player_two = generate_random_deck(num_cards, num_qubits)

    # Generate target bitstrings
    self.num_target_bitstrings = num_bitstrings_per_player
    self.bitstrings_player_one = set()
    while len(self.bitstrings_player_one) < self.num_target_bitstrings:
      self.bitstrings_player_one.add(generate_bitstring(self.num_qubits))
    self.bitstrings_player_two = set()
    while len(self.bitstrings_player_two) < self.num_target_bitstrings:
      the_random_bitstring = generate_bitstring(self.num_qubits)
      if the_random_bitstring in self.bitstrings_player_one:
        continue
      self.bitstrings_player_two.add(generate_bitstring(self.num_qubits))

    # Decide who goes first
    self.turn = random.choice([1, 2])

  def before_game_loop(self):
    """Prints information for the players before the game starts"""
    # This game really needs a better name
    print("Cards for player 1:", self.deck_player_one)
    print("Cards for player 2:", self.deck_player_two)
    print("Flipping coin to determine who goes first...")
    print(f"The first turn goes to: Player {self.turn}")
    time.sleep(1)
    clear_terminal()
    statevector_data = self.get_statevector_data()
    plot_statevector_data(statevector_data)

  def apply_grover_oracle_with_planted_sol(self, bitstring: str):
    """
    Applies a Grover oracle with the planted solution 
    corresponding to bitstring
    :param bitstring: The bitstring to perform an iteration of Grover for
    """
    # Apply X gates when you want that certain bit to be 0
    for i in range(len(bitstring)):
        index = len(bitstring) - 1 - i
        if bitstring[index] == 0:
            self.qc.x(self.q[index])
    self.qc.mcp(np.pi,self.q[1:],self.q[0])
    for i in range(len(bitstring)):
        index = len(bitstring) - 1 - i
        if bitstring[index] == 0:
            self.qc.x(self.q[index])

  def diffusion_operator(self):
    self.qc.h(self.q)
    self.qc.x(self.q)
    self.qc.mcp(np.pi,self.q[1:],self.q[0])
    self.qc.x(self.q)
    self.qc.h(self.q)

  def apply_move(self, card_played: Card):
    """
    Applies the move on a card played by the player.
    :param card_played: Card that the player uses on a turn.
    """
    qubit_numbers = []
    qubit_numbers = get_n_qubit_numbers(self.num_qubits,
                                        MOVE_TO_NUM_QUBITS[card_played.operation_name])

    match card_played.operation_name:
      case 'X':
        self.qc.x(self.q[qubit_numbers[0]])
      case 'Y':
        self.qc.y(self.q[qubit_numbers[0]])
      case 'Z':
        self.qc.z(self.q[qubit_numbers[0]])
      case 'H':
        self.qc.h(self.q[qubit_numbers[0]])
      case 'CX':
        self.qc.cx(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]])
      case 'CZ':
        self.qc.cz(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]])
      case 'CCX':
        self.qc.ccx(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]], self.q[qubit_numbers[2]])
      case 'CCZ':
        self.qc.ccz(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]], self.q[qubit_numbers[2]])
      case 'S':
        self.qc.s(self.q[qubit_numbers[0]])
      case 'S_dag':
        self.qc.sdg(self.q[qubit_numbers[0]])
      case 'T':
        self.qc.t(self.q[qubit_numbers[0]])
      case 'T_dag':
        self.qc.tdg(self.q[qubit_numbers[0]])
      case 'SWAP':
        self.qc.swap(self.q[qubit_numbers[0]], self.q[qubit_numbers[1]])
      case 'DIFFUSION':
        self.diffusion_operator()
      case 'RESET':
        self.qc.reset(self.q[qubit_numbers[0]])
        if random.random() < 0.5:
          self.qc.x(self.q[qubit_numbers[0]])
      case 'GROVER':
        bitstring = get_grover_input(self.num_qubits)
        self.apply_grover_oracle_with_planted_sol(bitstring)
        self.diffusion_operator()


  def measure_output_end_of_game(self, num_shots=1024):
    """Measures output at the end of the game using num_shots shots"""
    for i in range(self.num_qubits):
      self.qc.measure(self.q[i], self.c[i])
    backend = Aer.get_backend('qasm_simulator')
    result = backend.run(self.qc, shots=num_shots).result()
    counts = result.get_counts()
    return counts

  def get_statevector_data(self):
    """Gets the statevector data after all the moves that have been applied
    thus far"""
    v_init = Statevector.from_label('0' * self.num_qubits)
    v_evolved = v_init.evolve(self.qc)
    return v_evolved.probabilities_dict()

  def print_statevector_data(self, statevector_data):
    """
    Prints entries of a statevector
    :param statevector_data: Vector representing a quantum state
    :type statevector_data: list
    """
    print(statevector_data)

  def get_target_bitstring_counts(self, counts):
    """
    Determines number of times that each player has one of their target
    bitstrings occur.
    :param counts: Number of occurrences of each measured outcome
    :type counts: dict
    :returns: tuple(int, int) with counts for each player
    """
    count_player_one = 0
    count_player_two = 0
    for key in counts:
      if key in self.bitstrings_player_one:
        count_player_one += counts[key]
      if key in self.bitstrings_player_two:
        count_player_two += counts[key]
    return (count_player_one, count_player_two)

  def get_winner(self, counts):
    """
    Determines winner of the game based on measurement counts of the circuit.
    The player who has one of their target bitstrings occur more times wins.
    :param counts: Number of occurrences of each measured outcome
    :type counts: dict
    :returns: 1 if player 1 wins, 2 if player 2 wins, 3 if tie.
    """
    count_player_one, count_player_two = self.get_target_bitstring_counts(counts)
    if count_player_one > count_player_two:
      return 1
    elif count_player_one < count_player_two:
      return 2
    else:
      return 3

  def handle_player_turn(self):
    # clear_output(wait=True)
    print(f"Turn: Player {self.turn}")
    current_player_deck = (self.deck_player_one if self.turn == 1 else self.deck_player_two)
    current_player_bistrings = (self.bitstrings_player_one if self.turn == 1
                                else self.bitstrings_player_two)
    print("Your bitstrings:", current_player_bistrings)
    print("Your cards:", current_player_deck)
    card_played = get_deck_operation(current_player_deck)
    # Apply unitary corresponding to card
    self.apply_move(card_played)
    statevector_data = self.get_statevector_data()
    plot_statevector_data(statevector_data)
    time.sleep(3)
    clear_terminal()

  def game_loop(self):
    loop_counter = self.num_cards_per_player
    while loop_counter > 0:
      print("Number of turns left:", loop_counter)
      loop_counter -= 1
      self.handle_player_turn()
      self.turn = 2 if self.turn == 1 else 1
      self.handle_player_turn()
      self.turn = 2 if self.turn == 1 else 1

  def end_of_game(self):
    """Measures circuit and prints who won or if there was a tie"""
    plt.ioff()
    print("Game has ended, measuring the circuit...")
    counts = self.measure_output_end_of_game()
    print("Counts =", counts)
    target_bitstring_counts = self.get_target_bitstring_counts(counts)
    target_bitstring_count_player_one = target_bitstring_counts[0]
    target_bitstring_count_player_two = target_bitstring_counts[1]
    print("# of times player 1 measured one of their target bitstrings:",
          target_bitstring_count_player_one)
    print("# of times player 2 measured one of their target bitstrings:",
          target_bitstring_count_player_two)
    if target_bitstring_count_player_one > target_bitstring_count_player_two:
      print("Player 1 wins!")
    elif target_bitstring_count_player_one < target_bitstring_count_player_two:
      print("Player 2 wins!")
    else:
      print("Game results in a tie.")

  def play_game(self):
    self.before_game_loop()
    self.game_loop()
    self.end_of_game()

if __name__ == '__main__':
    print_game_rules()
    ready_to_proceed = input()

    clear_terminal()

    num_qubits_to_play_on = 0
    while True:
      try:
        num_qubits_to_play_on = int(input("Enter number of qubits you want to play the game using:"))
        if not (3 <= num_qubits_to_play_on <= 5):
          print("Enter a positive integer between 3 and 5, inclusive.")
        else:
          break
      except ValueError:
        print("Enter a positive integer between 3 and 5, inclusive.")
    
    num_cards = 0
    while True:
      try:
        num_cards = int(input("Enter number of cards you want to play the game using:"))
        if not (3 <= num_qubits_to_play_on <= 5):
          print("You must play with at least one card.")
        else:
          break
      except ValueError:
        print("Enter a positive integer.")
    
    num_bitstrings = int(2 ** (num_qubits_to_play_on - 2))
    clear_terminal()

    # Play game
    game = Game(3, 5, 2)
    game.play_game()