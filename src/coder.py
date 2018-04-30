# Encode and decode byte/bit vectors. Chunk them in the right size, apply error recovery algorithms like
# Reed Salomon or Polar Code and "randomize" the byte/bit vectors to have P(0) = P(1) = 1/2 (Mult by huge
# prime + mod 256). + All the reverse operations.
