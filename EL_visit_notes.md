# Notes on Ed Large's visit

author: jorgeh
date: May 22, 2014


## Schedule

- 10:00am – 12:00am : (brief intro) specific questions about the model
- 12:00pm – 01:00pm : lunch
- 01:00pm – 03:30pm : general questions / research ideas
- 03:30pm – 04:00pm : coffee break
- 04:00pm – 05:45pm : talk
- 06:00pm – 07:30pm : dinner (attendees: Ed, Jonathan, Ge, Takako?, Rob, Luke, Madeleine, Me = 8)


## Questions/discussion

### GFNN Specifics

1.  What is the mechanism (idea) behind the aggregation of stimulus, internal and aff / eff connections ( "derivatives" in zdot.m and similar files). From the paper (eq. 6), I don't get why the nonlinearity.

    **Notes:**
    


1.  What is that sd/12 in resonances() inside connectMakeLog?

    **Notes:**
    Hack. The new version solves this.
        

1.  Why divide R by n1.per (oscillators per octave)?

    **Notes:**
    To keep constant sum
        

1.  What does the complex kernel mean in the connections? I understand they are phase shifts being applied to z, but don't understand why is that desirable / needed and how does it relate to perception.

    **Notes:**
    Interesting!!!
        

1.  What is the idea behind the nml nonlinearity in zdot (or zdotH, etc.)? 

    **Notes:**
    
        

     1. Why is there a hardcoded m=0.4? 

        **Notes:**
    
        

     1. How was it found?

        **Notes:**
    
        

1.  Why the distinction between afferent and efferent connections? I know one goes "up" while the other goes "down", but from the point of view of a net, that's transparent, right?

    **Notes:**
    
        

1.  Why the frequency scaling in zdotH? (\sqrt{f})

    **Notes:**
    
        

1.  Why do I need to scale down the input (e.g. 0.5 factor)? This seems related to \epsilon (see second paragraph, second column p. 907). What is the trade-off?

    **Notes:**
    
        

1.  How and why the phase seems to work for beat tracking?

    **Notes:**
    
        

1.  How to make networks react quicker?

    **Notes:**
    Look into different frequency
    

     1. Why do they always use the average of the TF?

        **Notes:**
    
        

1.  Opinion on onset detection function as input to the model? Any objections with the idea of using an onset detection function?

    **Notes:**
    It is tightly coupled with the nature of the dynamical system. Maybe is not wise to think of it as "get the best onset detector there is" and then "get the best beat tracker". They are very intertwined.
        

1.  What about the problem of extreme sensitivity to stimulus level? There seems to be a noticeable difference between accented vs. non-accented stimulus.

    **Notes:**
    
        

1.  Why are the internal 'rhythmic' connections necessary? If I'm correct, a GFNN without it is also able to generate these simple harmonic ratios

    **Notes:**
    
        

1.  How can I gain some intuition on how params affect the model?

    **Notes:**
    
        

1.  Do you have a working beat tracker? 

    **Notes:**
    
        

     1. How can the phase issue be solved? 

        **Notes:**
    
        

     1. Has it been tested on real audio mixtures?

        **Notes:**
    
        

     1. How can the continuity issue (tracking) be solved? 

        **Notes:**
    
        

     1. What about interpolation?

        **Notes:**
    
        

1. How do you think about the second layer? I think here is where I think I could implement the binary/ternary meter distinction.

    **Notes:**
    
        




### General conversation / discussion

1.  Are there any issues with publishing the pyGFNN source code?

    **Notes:**
    
        

1.  Patents/licensing issues?

    **Notes:**
    

        
1.  How to handle time-signature ambiguity? How about poly-meters?

    **Notes:**
    
  

1.  Groove: is the "groove problem" trackable?

    **Notes:**
    
    In his view, the most metrial
        

1.  Any thoughts on how to make the 2D TF repr into a 3D repr?

    **Notes:**
    
        

1.  Idea: a second input source (instead of source separated stimuli) could be a belief signal (e.g. the experiment where listeners where asked to imagine a binary or ternary rhythm)

    **Notes:**
    
        

1.  What is his opinion of the idea of sensory-motor coupling for beat tracking (e.g. Todd's model)?

    **Notes:**
    
        

1.  Do you see any potential connection with ML (deep nets)?

    **Notes:**    
        




1.  How could I help you?

    **Notes:**
    
        
   
    1. Anything you would like to visualize?

    **Notes:**
    
        

    1. Some experiment you have thought about?

    **Notes:**
    
        

1.  How about handling multiple inputs (separated sources?)

    **Notes:**
    
        

1.  QBT dataset

    **Notes:**    
        


1.  What is the status of Circular Logic? Does it use GFNNs for beat tracking?

    **Notes:**
    
        


## The talk

### Notes

- epsilon controls the amount of Nonlinearity in the system when holding x(t) fixed
- 


### Questions

- For the rhythm model, the 2nd layer is the "motor network". Does it make sense to connect to a plant model? 
- Are "people's mistakes" really mistakes?
- 



# OThers

- He will send me a few chapters of his current book.
- He'd appreciate some feedback

- Phase is neutrally stable