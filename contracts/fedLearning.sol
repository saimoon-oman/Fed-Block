// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;
pragma experimental ABIEncoderV2;

import "./flockie.sol";

contract FedLearning{
    string server;
    // string public test = "Hello";
    uint voterCount;
    uint clientCount;
    mapping(address => string) clientWeights;

    mapping(address => bool) public isRegistered;
    mapping(address => bool) public hasVoted;

    uint update; 
    uint noUpdate;

    constructor() {
        // Initilizing default values
        clientCount = 0;
        voterCount = 0;
        update = 0;
        noUpdate = 0;
    }

    function sendWeights(address x, string memory y) public {
        clientWeights[x] = y;
        isRegistered[x] = true;
        hasVoted[x] = false;
        clientCount += 1;
    }

    function setServer(string memory serverHash) public returns(bool){
        server = serverHash;
        return true;
    }


    function getServer() public view returns(string memory){
        return server;
        // return test;
    }

    function getWeights(address a) public view returns(string memory){
        // assert(msg.sender == a);
        return clientWeights[a];
    }    

    function vote(address voter, uint acc) public payable {
        assert(isRegistered[voter]==true); //voter == msg.sender
        if(acc > 7500000){
            update += 1;
        }
        else{
            noUpdate += 1;
        }
        hasVoted[voter] = true;
        voterCount += 1;
    }

    function getVoteUpdate() public view returns(bool){
        assert(update + noUpdate == voterCount);
        if(update > noUpdate){
            return true;
        }
        return false;
        // return true;
    }

    // function accuracyChecker(string memory z, uint acc) public returns (bool){
    //     if(acc > 8000000){
    //         server = z;
    //         return true;
    //     }
    //     return false;
    // }    


}