const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {

    if (image.length > 1000000) {
      throw new InputError('Payload content length greater than maximum allowed: 1000000');
    }
    
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    
    const prediction = model.predict(tensor);
    const score = await prediction.data();
    
    
    const cancerProbability = score[0]; 
    const confidenceScore = cancerProbability * 100;

    const label = cancerProbability > 0.5 ? 'Cancer' : 'Non-cancer'; 

    let suggestion;
    if (label === 'Cancer') {
      suggestion = "Segera periksa ke dokter!";
    } else {
      suggestion = "Tetap jaga kesehatan!";
    }
    return { confidenceScore, label, suggestion };
  }
   catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClassification;
