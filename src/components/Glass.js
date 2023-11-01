import './Glass.css'
import { useState } from 'react'
import axios from 'axios'
import parse from "html-react-parser";
import {abi_fedLearning,contractAddress_fedLearning, abi_flockie, contractAddress_flockie, FLK_wolf, FLK_elephant, FLK_tiger} from './sc_config'
import Web3 from 'web3'
import UploadPage from './UploadPage'

function Glass() {
  const [x, setx] = useState('')
  // const [first, setfirst] = useState('')
  const [train, setTrain] = useState(false)
  const [approve, setApprove] = useState(false)
  const [server, setServer] = useState(false)
  const [update, setUpdate] = useState(false)
  
  const web3 = new Web3("http://localhost:7545")
  const fedLearning = new web3.eth.Contract(abi_fedLearning,contractAddress_fedLearning)
  const flockie = new web3.eth.Contract(abi_flockie, contractAddress_flockie) 


  const handleSubmit = async (e) => {
    e.preventDefault()      // Prevents the default form submission behavior to handle data via code.
    var accounts = []         // Initializes an empty array 'accounts' to store Ethereum accounts.
    const account_addr = await web3.eth.getAccounts()           // Retrieves Ethereum accounts available through Web3.
    console.log("All Ethemereum Accounts: ")
    console.log(account_addr)             // Logs the array of Ethereum accounts to the console.
    for(var element in account_addr){       // Loops through the 'account_addr' array and adds each account to the 'accounts' array.
      accounts.push(account_addr[element])
    }
    const serverhash = await axios.get('http://localhost:8080/initiallySetServerWeights')
    console.log(serverhash.data.hash)
    await fedLearning.methods.setServer(serverhash.data.hash).send({from:accounts[0], gas: 3000000})
    const s_hash = []
    await fedLearning.methods.getServer().call().then((server)=>{
      // console.log(data)
      s_hash.push(server)
      console.log(s_hash)
    }); 
    const params = { x }    // Creates an object with no. of client which is x
    const data = await axios.post('http://localhost:8080/train', params)    // Sends a POST request to 'http://localhost:8080/train' endpoint with the 'params' object.   Receives the response data returned from the server.
    
    console.log("Accounts List: ",accounts[1])      // Logs the second account from the 'accounts' array to the console.
    for (let i = 1; i <= x; i++) {
      fedLearning.methods.sendWeights(accounts[i], data.data.data[`client${i-1}`]).send({from:accounts[i], gas: 3000000}); // Calls the 'sendWeights' method of the 'fedLearning' contract to send client0's data to the second Ethereum account.  // The 'from' parameter specifies the sender's account and 'gas' is the gas limit for the transaction.
    }
    // fedLearning.methods.sendWeights(accounts[1], data.data.data.client0).send({from:accounts[0], gas: 3000000}); // Calls the 'sendWeights' method of the 'fedLearning' contract to send client0's data to the second Ethereum account.  // The 'from' parameter specifies the sender's account and 'gas' is the gas limit for the transaction.
    // fedLearning.methods.sendWeights(accounts[2], data.data.data.client1).send({from:accounts[0], gas: 3000000}); // Similarly, sends client1's data to the third Ethereum account.
    // fedLearning.methods.sendWeights(accounts[3], data.data.data.client2).send({from:accounts[0], gas: 3000000}); // Sends client2's data to the fourth Ethereum account.
    setTrain(true)   // Updates the state variable 'train' to true, indicating that the training process is completed.
  }

  const handleAggregate = async(e) => {
    e.preventDefault()      // Prevents the default form submission behavior to handle data via code.

    const accounts = []
    const account_addr = await web3.eth.getAccounts()
    for(const element in account_addr){
      accounts.push(account_addr[element])
    }   // Retrieves Ethereum accounts using Web3 and stores them in the 'accounts' array.

    const payload = []   // Initializes an empty array 'payload' to store data.
    
    for (let i = 1; i <= x; i++) {
      await fedLearning.methods.getWeights(accounts[i]).call().then((data)=>{
        // console.log(data)
        payload.push(data)
      });   // Retrieves weights for account 1 from the 'fedLearning' contract and adds it to the 'payload' array.  
      console.log("Get data from address ", accounts[i])
      console.log("data: ", payload[i-1])
    }
    // await fedLearning.methods.getWeights(accounts[1]).call().then((data)=>{
    //   // console.log(data)
    //   payload.push(data)
    // });   // Retrieves weights for account 1 from the 'fedLearning' contract and adds it to the 'payload' array.
    // // console.log("aggregation1")
    // // console.log("here")
    // console.log("Get data from address ", accounts[1])
    // console.log("data: ", payload[0])
    // await fedLearning.methods.getWeights(accounts[2]).call().then((data) => {
    //   payload.push(data)
    // });     // Similarly, fetches weights for accounts 2 and 3 and adds them to the 'payload' array.
    // console.log("Get data from address ", accounts[2])
    // console.log("data: ", payload[1])
    // await fedLearning.methods.getWeights(accounts[3]).call().then((data) => {
    //   payload.push(data)
    // });
    // console.log("Get data from address ", accounts[3])
    // console.log("data: ", payload[2])
    
    //Mint NFT
    // Mint NFTs for three accounts using 'flockie' contract.
    // await flockie.methods.mintNFT(accounts[7], FLK_wolf).send({from:accounts[0], gas: 3000000});
    // await flockie.methods.mintNFT(accounts[8], FLK_elephant).send({from:accounts[0], gas: 3000000});
    // for (let i = 0; i < x; i++) {
    //   await flockie.methods.mintNFT(accounts[9-i], FLK_tiger).send({from:accounts[0], gas: 3000000});
    // }
    // Sends NFTs to accounts 4, 5, and 6 with respective token identifiers.

    const data_agg = await axios.post('http://localhost:8080/aggregate', payload)    // Posts the 'payload' data to 'http://localhost:8080/aggregate' and gets the aggregated data.
    
    setServer(true)   // Sets the 'server' state to true, indicating the server has received aggregated data
    console.log("HELLLLOOOOOOO")
    // Votes on the accuracy of the aggregated data for the respective accounts using the 'flockie' contract

    for (let i = 1; i <= x; i++) {
      await fedLearning.methods.vote(accounts[i], data_agg.data.data.accuracy[i-1]).send({from:accounts[i], gas: 3000000});  
    }

    const upd = await fedLearning.methods.getVoteUpdate().call()  // Fetches update information from the 'flockie' contract

    if(upd){
      // If an update is available, set the server hash on the 'fedLearning' contract
      var data = await fedLearning.methods.setServer(data_agg.data.data.hash).send({from:accounts[0], gas: 3000000})
      if (data) {
        setUpdate(true)
        setApprove(data)
      }
      // await fedLearning.methods.setServer(data_agg.data.data.hash).call().then((data) => {
      //   setUpdate(true)
      //   setApprove(data)  // Updates the 'update' state and approves the data from the 'fedLearning' contract
      // })
    }
  }

  // const reset = () => {
  //   setx('')
  //   setfirst('')
  // }

 
  return (
    <>
    {!train &&
    <div className="glass">
      <form onSubmit={(e) => handleSubmit(e)} className="glass__form">
        <h4>Train Clients</h4>
        <div className="glass__form__group">
          <input
            id="Client_count"
            className="glass__form__input"
            placeholder="Number of Clients"
            required
            autoFocus
            min="2"
            // max="1"
            pattern="[0-9]{0,1}"
            title="Client count"
            type="number"
            value={x}
            onChange={(e) => setx(e.target.value)}
          />
        </div>

        {/* <div className="glass__form__group">
          <input
            id="bsc"
            className="glass__form__input"
            placeholder="True or False"
            required
            // min="0"
            // max="5"
            type="bool"
            title="First time?"
            // pattern="[0-9]+([\.,][0-9]+)?"
            // step="0.01"
            value={first}
            onChange={(e) => setfirst(e.target.value)}
          />
        </div> */}

        <div className="glass__form__group">
          <button type="submit" className="glass__form__btn">
            Train Clients and Upload Gradients
          </button>
        </div>
      </form>
    </div>}

    {train && !server &&
    <div className="glass">
      <form onSubmit={(e) => handleAggregate(e)} className="glass__form">
        <h4>Aggregate Clients</h4>
        <div className="glass__form__group">
          <button type="submit" className="glass__form__btn">
            Approve
          </button>
        </div>
      </form>
    </div>}
    
    {train && server && approve &&
      <div className="glass">
          <h4>Congratulations, your model has been approved!</h4>
      </div>}

    

    {train && server && !approve &&
    <div className="glass">
        <h4>Please improve your model and try again!</h4>
    </div>}

    </>)
}

export default Glass