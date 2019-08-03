class Bird {

  constructor(idn) {
    this.id = idn
    this.brain = new NeuralNetwork(3,5,2)
    this.init()
    colorMode(HSB,100)
    this.color = color(Math.random()*100,100,100)
  }

  init(){
    this.score = 0
    this.isalive = 1
    this.y = height / 2
    this.x = 64
    this.gravity = 0.6
    this.lift = -16
    this.velocity = 0
    this.desicion = 0
    this.counter = 0
  }

  show(){
    fill(this.color)
    ellipse(this.x, this.y, 32, 32 )
  }
  
  goUp(){
    this.velocity += this.lift
  }

  look(){
    // [distance from obst, vert dis from obst, speed]

    this.data_in = [Math.min(obstacles[0].x-64)/210,(obstacles[0].gapStart+60-this.y)/100,this.velocity]
  }

  decide(){
    if (this.counter%10==0){
      this.look()
      this.desicionx = this.brain.predict(this.data_in)
      if(this.brain.predict(this.data_in)[0]>this.brain.predict(this.data_in)[1]){
        this.desicion = 1
      }
    }
    else{
      this.desicion = 0
    }
  }
  
  update(){
    this.velocity += this.gravity
    this.velocity *= 0.9
    this.y += this.velocity
    
    if (this.y > height) {
      this.y = height
      this.velocity = 0
    }
    
    if (this.y < 0) {
      this.y = 0
      this.velocity = 0
    }
  }

  mutate(){
    function fn(x) {
      if (random(1) < 0.05) {
        let offset = randomGaussian() * 0.5;
        let newx = x + offset;
        return newx;
      }
      return x;
    }

    let ih = this.brain.input_weights.dataSync().map(fn);
    let ih_shape = this.brain.input_weights.shape;
    this.brain.input_weights.dispose();
    this.brain.input_weights = tf.tensor(ih, ih_shape);
    
    let ho = this.brain.output_weights.dataSync().map(fn);
    let ho_shape = this.brain.output_weights.shape;
    this.brain.output_weights.dispose();
    this.brain.output_weights = tf.tensor(ho, ho_shape);
  }

  crossover(partner){
    let parentA_in_dna = this.brain.input_weights.dataSync();
    let parentA_out_dna = this.brain.output_weights.dataSync();
    let parentB_in_dna = partner.brain.input_weights.dataSync();
    let parentB_out_dna = partner.brain.output_weights.dataSync();

    let mid = Math.floor(Math.random() * parentA_in_dna.length);
    let child_in_dna = [...parentA_in_dna.slice(0, mid), ...parentB_in_dna.slice(mid, parentB_in_dna.length)];    
    let child_out_dna = [...parentA_out_dna.slice(0, mid), ...parentB_out_dna.slice(mid, parentB_out_dna.length)];

    let child = this.clone();
    let input_shape = this.brain.input_weights.shape;
    let output_shape = this.brain.output_weights.shape;
    
    child.brain.dispose();

    child.brain.input_weights = tf.tensor(child_in_dna, input_shape);
    child.brain.output_weights = tf.tensor(child_out_dna, output_shape);
    
    return child;
  }

  kill(){
    this.brain.dispose();
  }

  clone(){
    let params = Object.assign({}, this.params);
    let new_person = new Bird(params);
    new_person.brain.dispose();
    new_person.brain = this.brain.clone();
    return new_person;
  }
}