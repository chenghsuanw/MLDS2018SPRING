import numpy as np
import pickle

class Data(object):
	def __init__(self):
		self.train_current = 0
		self.val_current = 0
		self.encoder_input = None
		self.decoder_input = None
		self.decoder_output = None
		self.val_encoder_input = None
		self.val_decoder_input = None
		self.val_decoder_output = None

	def tokenize(self, data, w2i, max_length):
		print('Tokenizing data...')

		encoder_input, decoder_input, decoder_output = [], [], []

		for i in range(data.shape[0]):
			feat = data[i][0]
			caption = data[i][1]

			encoder_input.append(feat)
			d_in, d_out = [], []
			caption_words = caption.split()

			d_in.append(w2i['<BOS>'])
			for i, word in enumerate(caption_words):
				if i < max_length-1:
					if word in w2i:
						d_in.append(w2i[word])
						d_out.append(w2i[word])
					else:
						d_in.append(w2i['<UNK>'])
						d_out.append(w2i['<UNK>'])
				else:
					break
			d_out.append(w2i['<EOS>'])

			# padding
			while len(d_in) < max_length:
				d_in.append(w2i['<PAD>'])
				d_out.append(w2i['<PAD>'])

			decoder_input.append(d_in)
			decoder_output.append(d_out)

		self.encoder_input = encoder_input[:20000]
		self.decoder_input = decoder_input[:20000]
		self.decoder_output = decoder_output[:20000]
		self.val_encoder_input = encoder_input[20000:]
		self.val_decoder_input = decoder_input[20000:]
		self.val_decoder_output = decoder_output[20000:]

		print('Finished tokenizing data')

	def next_batch(self, size, val=False):
		if not val:
			if self.train_current + size < len(self.encoder_input):
				encoder_input = np.array(self.encoder_input[self.train_current:self.train_current+size])
				decoder_input = np.array(self.decoder_input[self.train_current:self.train_current+size])
				decoder_output = np.array(self.decoder_output[self.train_current:self.train_current+size])
				self.train_current += size
			else:
				encoder_input = np.array(self.encoder_input[self.train_current:])
				decoder_input = np.array(self.decoder_input[self.train_current:])
				decoder_output = np.array(self.decoder_output[self.train_current:])
				self.train_current = 0
		else:
			if self.val_current + size < len(self.val_encoder_input):
				encoder_input = np.array(self.val_encoder_input[self.val_current:self.val_current+size])
				decoder_input = np.array(self.val_decoder_input[self.val_current:self.val_current+size])
				decoder_output = np.array(self.val_decoder_output[self.val_current:self.val_current+size])
				self.val_current += size
			else:
				encoder_input = np.array(self.val_encoder_input[self.val_current:])
				decoder_input = np.array(self.val_decoder_input[self.val_current:])
				decoder_output = np.array(self.val_decoder_output[self.val_current:])
				self.val_current = 0

		return encoder_input, decoder_input, decoder_output
		

