
(succ 0 1) (succ 1 2) (succ 2 3) (succ 3 4) (succ 4 5) (succ 5 6) (succ 6 7) (succ 7 8) (succ 8 9) (succ 9 10) (succ 10 11) (succ 11 12) (succ 12 13) (succ 13 14) (succ 14 15) (succ 15 16) (succ 16 17) (succ 17 18) (succ 18 19) (succ 19 20) (succ 20 21) (succ 21 22) (succ 22 23) (succ 23 24) (succ 24 25) (succ 25 26) (succ 26 27) (succ 27 28) (succ 28 29) (succ 29 30) 
(succ 30 31) (succ 31 32) (succ 32 33) (succ 33 34) (succ 34 35) (succ 35 36) (succ 36 37) (succ 37 38) (succ 38 39) (succ 39 40) (succ 40 41) (succ 41 42) (succ 42 43) (succ 43 44) (succ 44 45) (succ 45 46) (succ 46 47) (succ 47 48) (succ 48 49) (succ 49 50) (succ 50 51)
;            succ
;             v
(exec (IC 0 1 0)
               (, (exec (IC $x $y $a) $sp $st) (succ $a $ab)
                  ((step $x) $p $t))
               (, (exec (IC $y $x $ab) $sp $st)
                  (exec (R $x) $p $t)
                  (step (IC 0 1 0) $sp $st))) ;; adding its instance for later execution

((step 0)
      (, (petri (? $channel $payload $body))
         (petri (! $channel $payload)) )
      (, (petri $body)))

((step 1)
      (, (petri (| $lprocess $rprocess)))
      (, (petri $lprocess)
         (petri $rprocess)))

;; run generation :- run all steps for one generation
((run-gen)
       (, ((step_r $name) $p1 $t1))
       (, (exec $name $p1 $t1))
)

;; evaluate individual fitness
 ((step_r A)
     (, (petri (Cons $h $t))
        (step (IC 0 1 0) $p1 $t1))
     (, (petri (! (fitness (Cons $h $t) ) (Cons $h $t)))
        (exec (IC 0 1 0) $p1 $t1)))

;; select individual if score >= threshold
 ((step_r B)
       (, (petri (! (Cons $h $t) $score))
          (step (IC 0 1 0) $p1 $t1))
       (, (petri (! (filter-one-element RESULT_1) ( ((Cons $h $t) $score) (S (S Z))) ) )
          (exec (IC 0 1 0) $p1 $t1)))


;; mutate evaluated individual
((step_r C )
       (, (petri (! selected (Cons $h $t)) )
          (step (IC 0 1 0) $p1 $t1))
       (, (petri (! (mutate New_ind ) ( (Cons $h $t) (S (S Z)))))   ;; TODO random mutation index
          (petri (! (mutate New_ind ) ( (Cons $h $t) Z)))
          (exec (IC 0 1 0) $p1 $t1)))

;; Crossover two evaluated individuals
((step_r D )
       (, (petri (! selected (Cons $h1 $t1)) )
          (petri (! selected (Cons $h2 $t2)) )
          (step (IC 0 1 0) $ptt $tmp))
       (, (petri (! (crossover New_siblings ) ( (S (S Z)) (Cons $h1 $t1) (Cons $h2 $t2))) ) ;; TODO random crossover point
          (exec (IC 0 1 0) $ptt $tmp)))

;; clean up for next generation
;;      And add the new population and continue the process 

((step_r E)
       (, (petri (Cons $h1 $t1))
          (petri (! (Cons $h2 $t2) $score))
          (petri (! selected (Cons $h3 $t3)))
          (petri (! New_ind (Cons $h4 $t4)))
          (petri (! New_siblings ((Cons $h5 $t5) (Cons $h6 $t6)))))
       (O (- (petri (Cons $h1 $t1)))
          (- (petri (! (Cons $h2 $t2) $score)))
          (- (petri (! selected (Cons $h3 $t3))))
          (- (petri (! New_ind (Cons $h4 $t4))))
          (- (petri (! New_siblings ((Cons $h5 $t5) (Cons $h6 $t6)))))
          (+ (petri (Cons $h4 $t4)))
          (+ (petri (Cons $h5 $t5)))
          (+ (petri (Cons $h6 $t6)))))

(Gen 0 1) (Gen 1 2) (Gen 2 3) (Gen 3 4) (Gen 4 5) 
;; Main loop :-
;;         run for 5 generations
(exec (Main 0)
       (, (exec (Main $a) $p $t )
           ((run-gen) $p1 $t1) (Gen $a $b))
       (, (exec run-gen $p1 $t1)
          (exec (Main $b) $p $t )
         (Gen $a $b)))


;; intial popluation

(petri (Cons 0 (Cons 1 (Cons 0 (Cons 1 (Cons 1 Nil))))))
(petri (Cons 0 (Cons 0 (Cons 1 (Cons 1 (Cons 1 Nil))))))
(petri (Cons 1 (Cons 1 (Cons 0 (Cons 1 (Cons 1 Nil))))))
(petri (Cons 0 (Cons 0 (Cons 0 (Cons 0 (Cons 0 Nil))))))
(petri (Cons 1 (Cons 1 (Cons 1 (Cons 1 (Cons 0 Nil))))))
(petri (Cons 1 (Cons 0 (Cons 0 (Cons 0 (Cons 0 Nil))))))    

;; calculate fitness

(petri (? (fitness $ret) Nil (! $ret Z)))

(petri
  (? (fitness $ret) (Cons 0 $tail)
      (| (! (fitness (PN $tail)) $tail)
         (? (PN $tail) $z (! $ret $z))))
)

(petri
  (? (fitness $ret) (Cons 1 $tail)
      (| (! (fitness (PN $tail)) $tail)
         (? (PN $tail) $z (! $ret (S $z)))))
)


;;select if score >= threshold
(petri (? (filter-one-element $ret) (($individual $score) $k)
        (|  (! (is-greater $individual ) ($score $k))
            (|  (? $individual True (! selected $individual))
                (? $individual False (! Nil) )
            )
        )
))

;; compare two peano numbers
(petri
  (? (is-greater $ret) ((S $a) (S $b))
      (| (! (is-greater ($ret sub)) ($a $b))
         (? ($ret sub) $result (! $ret $result))))
)

(petri (? (is-greater $ret) ((S $a) Z) (! $ret True)))
(petri (? (is-greater $ret) (Z $any) (! $ret False)))



;; mutation of bits in the list
(petri
  (? (mutate $ret) ((Cons $head $tail) (S $k))
      (| (! (mutate ($ret sub)) ($tail $k))
         (? ($ret sub) $mutated_tail (! $ret (Cons $head $mutated_tail)))))
)

(petri (? (mutate $ret) ((Cons 0 $tail) Z) (! $ret (Cons 1 $tail))))
(petri (? (mutate $ret) ((Cons 1 $tail) Z) (! $ret (Cons 0 $tail))))
(petri (? (mutate $ret) (Nil $any_index) (! $ret Nil)))



;; crossover of two lists
(petri
  (? (crossover $ret) ((S $n) (Cons $h1 $t1) (Cons $h2 $t2))
      (| (! (crossover (PN $n $t1 $t2)) ($n $t1 $t2))
         (? (PN $n $t1 $t2) ($z1 $z2)
              (! $ret ((Cons $h1 $z1) (Cons $h2 $z2))))))
)

(petri
  (? (crossover $ret) ((S Z) (Cons $h1 $t1) (Cons $h2 $t2))
      (! $ret ((Cons $h1 $t2) (Cons $h2 $t1))))
)

(petri (? (crossover $ret) (Z $l1 $l2) (! $ret ($l2 $l1))))


