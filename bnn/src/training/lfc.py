import lasagne
import quantization as q
import quantized_net as qn

def genLfc(input, num_outputs, learning_parameters):
    # A function to generate the lfc network topology which matches the overlay for the Pynq board.
    # WARNING: If you change this file, it's likely the resultant weights will not fit on the Pynq overlay.
    if num_outputs < 1 or num_outputs > 64:
        error("num_outputs should be in the range of 1 to 64.")
    num_units = 1024
    n_hidden_layers = 3
    if learning_parameters.activation_bits == 1:
        act_quant = q.QuantizationBinary()
    else:
        act_quant = q.QuantizationFixed(learning_parameters.activation_bits,
            learning_parameters.activation_bits - 2)
    activation = qn.FixedHardTanH(act_quant)
    if learning_parameters.weight_bits == 1:
        weight_quant = q.QuantizationBinary()
    else:
        weight_quant = q.QuantizationFixed(learning_parameters.weight_bits,
            learning_parameters.weight_bits - 2)
    W_LR_scale = learning_parameters.W_LR_scale
    epsilon = learning_parameters.epsilon
    alpha = learning_parameters.alpha
    dropout_in = learning_parameters.dropout_in
    dropout_hidden = learning_parameters.dropout_hidden

    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)
            
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = qn.DenseLayer(
                mlp, 
                quantization=weight_quant,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    
    mlp = qn.DenseLayer(
                mlp, 
                quantization=weight_quant,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_outputs)
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)
    return mlp